"""
Evaluate a Vision-Language captioning model on COCO with tangible metrics.

Includes:
- compute_caption_metrics(preds, refs, images, ...) -> dict of metrics
- evaluate_vlm_on_coco(...)                         -> runs generation + metrics end-to-end

Requires:
  pip install evaluate bert-score nltk torchmetrics open_clip_torch pycocoevalcap rouge_score
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# -------- Optional deps (graceful fallbacks) --------
_HAS_EVAL = True
try:
    import evaluate
except Exception:
    _HAS_EVAL = False

_HAS_COCO = True
try:
    from pycocoevalcap.cider.cider import Cider
except Exception:
    _HAS_COCO = False

_HAS_TM = True
try:
    from torchmetrics.multimodal import CLIPScore
except Exception:
    _HAS_TM = False

# rouge_score (fast, per-example)
_HAS_ROUGE_SCORE = True
try:
    from rouge_score import rouge_scorer
except Exception:
    _HAS_ROUGE_SCORE = False

# Optional: PIL->Tensor conversion without hard dependency
try:
    from torchvision.transforms.functional import pil_to_tensor
    _HAS_TV = True
except Exception:
    _HAS_TV = False


# -------------------- Utils --------------------
def _flatten_text(s: str) -> str:
    return " ".join(s.strip().split())


def _detect_cols(ds) -> Tuple[str, str, str]:
    cols = set(ds.column_names)
    img_col = "image" if "image" in cols else ("img" if "img" in cols else None)
    cap_col = "caption" if "caption" in cols else ("text" if "text" in cols else None)
    id_col = None
    for c in ("image_id", "id", "cocoid"):
        if c in cols:
            id_col = c
            break
    if not (img_col and cap_col and id_col):
        raise ValueError(
            f"Dataset columns not found. Have {sorted(cols)} "
            f"but need image/caption/id (one of: image|img, caption|text, image_id|id|cocoid)."
        )
    return img_col, cap_col, id_col


# -------------------- Metrics --------------------
def _compute_bleu_meteor_rouge(
    preds: List[str],
    refs: List[List[str]],
) -> Dict[str, Optional[float]]:
    """
    BLEU-4, METEOR, ROUGE-L (max-over-refs).
    Uses `evaluate` for BLEU+METEOR. For ROUGE-L we prefer `rouge_score` for speed.
    """
    out = {"bleu4": None, "meteor": None, "rougeL": None}

    if _HAS_EVAL:
        # BLEU-4 (handles multi-refs natively)
        bleu = evaluate.load("bleu")
        bleu_res = bleu.compute(predictions=preds, references=refs)
        out["bleu4"] = float(bleu_res["bleu"])  # 0..1

        # METEOR (supports multi-refs)
        meteor = evaluate.load("meteor")
        meteor_res = meteor.compute(predictions=preds, references=refs)
        out["meteor"] = float(meteor_res["meteor"])  # 0..1

    # ROUGE-L (max over refs per sample)
    if _HAS_ROUGE_SCORE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = []
        for p, rlist in zip(preds, refs):
            best_f = 0.0
            for r in rlist:
                score = scorer.score(r, p)  # (target, prediction) for rouge_score
                best_f = max(best_f, float(score["rougeL"].fmeasure))
            vals.append(best_f)
        out["rougeL"] = sum(vals) / max(1, len(vals))
    elif _HAS_EVAL:
        # fallback (slower): evaluate.rouge doesn't do multi-refs; approximate with max-over-refs
        rouge = evaluate.load("rouge")
        rouge_vals = []
        for p, rlist in zip(preds, refs):
            scores = rouge.compute(predictions=[p] * len(rlist), references=rlist)
            rouge_vals.append(float(scores["rougeL"]))
        out["rougeL"] = sum(rouge_vals) / max(1, len(rouge_vals))

    return out


def _compute_bertscore(preds: List[str], refs: List[List[str]]) -> Optional[float]:
    """
    BERTScore-F1 (max-over-refs per example), done in R passes (R: #refs) instead of N*R calls.
    """
    if not _HAS_EVAL:
        return None
    bs = evaluate.load("bertscore")

    n = len(preds)
    best = [0.0] * n
    # Find the maximum number of refs across samples
    rmax = max(len(r) for r in refs) if refs else 0

    for j in range(rmax):
        refs_j = [r[j] if j < len(r) else r[-1] for r in refs]  # pad with last ref
        res = bs.compute(predictions=preds, references=refs_j, lang="en")
        f1 = list(map(float, res["f1"]))
        best = [max(b, f) for b, f in zip(best, f1)]

    return sum(best) / max(1, n)


def _compute_cider(preds: List[str], refs: List[List[str]]) -> Optional[float]:
    """
    CIDEr-D using pycocoevalcap (standard COCO caption metric).
    """
    if not _HAS_COCO:
        return None
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def _pil_list_to_chw_tensors(images: List[Image.Image]) -> List[torch.Tensor]:
    if _HAS_TV:
        return [pil_to_tensor(img.convert("RGB")) for img in images]  # uint8 [C,H,W]
    out = []
    for img in images:
        arr = np.array(img.convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W], uint8
        out.append(t)
    return out


def _compute_clipscore(
    preds: List[str],
    images: Optional[List[Image.Image]],
    device: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",   # lighter than L/14; change if you want
    clip_batch_size: int = 8,
) -> Optional[float]:
    """
    CLIPScore via torchmetrics (reference-free image grounding).
    Stream updates in small batches to avoid CPU/GPU OOM.
    """
    if images is None or not _HAS_TM or len(images) == 0:
        return None

    img_tensors = _pil_list_to_chw_tensors(images)

    metric = CLIPScore(model_name_or_path=clip_model_name).to(device)
    # Stream to keep memory bounded
    with torch.no_grad():
        for i in range(0, len(img_tensors), clip_batch_size):
            ib = img_tensors[i : i + clip_batch_size]
            tb = preds[i : i + clip_batch_size]
            metric.update(ib, tb)
        s = metric.compute()
        metric.reset()

    # Free quickly
    try:
        del metric
    except Exception:
        pass

    return float(s.item() if hasattr(s, "item") else torch.as_tensor(s).float().item())


def compute_caption_metrics(
    preds: Sequence[str],
    refs: Sequence[Sequence[str]],
    images: Optional[Sequence[Image.Image]] = None,
    device: Optional[str] = None,
    *,
    compute_clipscore: bool = True,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_batch_size: int = 8,
    compute_bertscore: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute a suite of meaningful captioning metrics.

    Returns keys: bleu4, meteor, rougeL, bertscore_f1, cider, clipscore.
    Some values may be None if optional deps are missing or metrics are disabled.
    """
    preds = [_flatten_text(p) for p in preds]
    refs = [[_flatten_text(r) for r in rlist] for rlist in refs]
    if images is not None:
        images = [img.convert("RGB") for img in images]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out: Dict[str, Optional[float]] = {}
    out.update(_compute_bleu_meteor_rouge(list(preds), list(refs)))
    out["bertscore_f1"] = _compute_bertscore(list(preds), list(refs)) if compute_bertscore else None
    out["cider"] = _compute_cider(list(preds), list(refs))

    out["clipscore"] = None
    if compute_clipscore:
        out["clipscore"] = _compute_clipscore(
            list(preds),
            list(images) if images else None,
            device,
            clip_model_name=clip_model_name,
            clip_batch_size=clip_batch_size,
        )
    return out


# -------------------- Evaluation loop --------------------
def _unique_images_and_refs(
    ds,
) -> Tuple[List[int], Dict[int, List[str]], Dict[int, Image.Image]]:
    """
    Return:
      ids:         list of unique image ids
      refs_by_id:  id -> list[str] (all refs)
      image_by_id: id -> PIL.Image
    """
    img_col, cap_col, id_col = _detect_cols(ds)
    refs_by_id: Dict[int, List[str]] = defaultdict(list)
    image_by_id: Dict[int, Image.Image] = {}
    ids: List[int] = []

    for ex in ds:
        iid = int(ex[id_col])
        refs_by_id[iid].append(str(ex[cap_col]))
        if iid not in image_by_id:
            image_by_id[iid] = ex[img_col]
            ids.append(iid)
    return ids, refs_by_id, image_by_id


@torch.inference_mode()
def evaluate_vlm_on_coco(
    base_model_id: str,
    adapter_dir: str,
    split: str = "validation",
    dataset_id: str = "jxie/coco_captions",
    prompt: str = "Describe this image in detail.",
    max_images: int = 1000,
    batch_size: int = 4,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
    *,
    # metric toggles & perf knobs
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_batch_size: int = 8,
    enable_clipscore: bool = True,
    enable_bertscore: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Run generation on COCO and compute caption metrics.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data
    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception:
        fallback = "val" if split == "validation" else split
        ds = load_dataset(dataset_id, split=fallback)
    ids, refs_by_id, image_by_id = _unique_images_and_refs(ds)

    if max_images and max_images > 0:
        ids = ids[:max_images]

    # 2) Processor & model (+ LoRA if present)
    try:
        processor = AutoProcessor.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)

    model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
    )
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    except Exception:
        pass
    model.to(device).eval()

    tokenizer = getattr(processor, "tokenizer", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(model.config, "pad_token_id", None)
    eos_id = getattr(model.config, "eos_token_id", None) or getattr(tokenizer, "eos_token_id", None)

    # 3) Batched generation
    all_preds: List[str] = []
    all_refs: List[List[str]] = []
    all_imgs: List[Image.Image] = []

    def chunker(x, n):
        for i in range(0, len(x), n):
            yield x[i : i + n]

    for chunk in chunker(ids, batch_size):
        images = [image_by_id[i] for i in chunk]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            for _ in images
        ]
        prompts = [processor.apply_chat_template([m], add_generation_prompt=True) for m in messages]

        inputs = processor(
            text=prompts,
            images=images,
            padding=True,       # important for batching
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Use the *common padded length* for the batch; robust to left/right padding.
        prefix_len = inputs["input_ids"].shape[1]

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        for i in range(len(images)):
            new_tokens = gen_ids[i, prefix_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True) if tokenizer else ""
            all_preds.append(_flatten_text(text))
            all_refs.append([_flatten_text(r) for r in refs_by_id[chunk[i]]])
            all_imgs.append(images[i])

    # 4) Metrics (streaming CLIPScore to avoid OOM)
    metrics = compute_caption_metrics(
        all_preds,
        all_refs,
        images=all_imgs,
        device=device,
        compute_clipscore=enable_clipscore,
        clip_model_name=clip_model_name,
        clip_batch_size=clip_batch_size if torch.cuda.is_available() else max(1, min(4, clip_batch_size)),
        compute_bertscore=enable_bertscore,
    )

    # 5) Scorecard
    pretty = {
        "CIDEr (↑)": metrics["cider"],
        "CLIPScore (↑)": metrics["clipscore"],
        "BLEU-4 (↑)": metrics["bleu4"],
        "METEOR (↑)": metrics["meteor"],
        "ROUGE-L (↑)": metrics["rougeL"],
        "BERTScore-F1 (↑)": metrics["bertscore_f1"],
    }
    print("\n=== COCO Captioning — Scorecard ===")
    for k, v in pretty.items():
        if v is None:
            print(f"{k:16} : (not available)")
        else:
            if k.startswith(("BLEU", "METEOR", "ROUGE", "BERTScore")):
                print(f"{k:16} : {100.0 * float(v):.2f}")
            else:
                print(f"{k:16} : {float(v):.3f}")
    print(f"Evaluated {len(all_preds)} images from split='{split}'.\n")

    return metrics


# -------------------- CLI --------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--adapter_dir", default="outputs/smolvlm-coco-lora")
    ap.add_argument("--dataset_id", default="jxie/coco_captions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt", default="Describe this image in detail.")
    ap.add_argument("--max_images", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    # Optional metric/perf knobs
    ap.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_batch_size", type=int, default=8)
    ap.add_argument("--no_clipscore", action="store_true")
    ap.add_argument("--no_bertscore", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    evaluate_vlm_on_coco(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        split=args.split,
        dataset_id=args.dataset_id,
        prompt=args.prompt,
        max_images=args.max_images,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_model_name=args.clip_model_name,
        clip_batch_size=args.clip_batch_size,
        enable_clipscore=not args.no_clipscore,
        enable_bertscore=not args.no_bertscore,
    )

    # Optional: also run validation split
    evaluate_vlm_on_coco(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        split="validation",
        dataset_id=args.dataset_id,
        prompt=args.prompt,
        max_images=args.max_images,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_model_name=args.clip_model_name,
        clip_batch_size=args.clip_batch_size,
        enable_clipscore=not args.no_clipscore,
        enable_bertscore=not args.no_bertscore,
    )
