import base64
import io
import json
import os
from typing import Any, Dict, Tuple, List

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _maybe_bf16():
    if DEVICE == "cuda":
        # BF16 safe default on recent GPUs, fallback eager attn if flash-attn not present
        return torch.bfloat16
    return torch.float32

def _load_image_from_payload(d: Dict[str, Any]) -> Image.Image:
    if "image" in d:  # base64 string
        raw = base64.b64decode(d["image"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if "image_url" in d:
        # use transformers.image_utils to robustly fetch remote images
        return load_image(d["image_url"]).convert("RGB")
    raise ValueError("Request must include either 'image' (base64) or 'image_url'.")

def _build_messages(prompt: str, num_images: int) -> List[Dict[str, Any]]:
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]

# ===== Required by SageMaker Inference Toolkit (HuggingFaceModel) =====
def model_fn(model_dir: str):
    """
    Load the base model + apply LoRA from model_dir (the SM_MODEL_DIR tar contents).
    BASE_MODEL_ID can be overridden via environment variable.
    """
    base_model_id = os.environ.get("BASE_MODEL_ID", "HuggingFaceTB/SmolVLM-Instruct")
    image_longest_edge = os.environ.get("IMAGE_LONGEST_EDGE")  # optional int

    processor_kwargs = {}
    if image_longest_edge:
        try:
            processor_kwargs["size"] = {"longest_edge": int(image_longest_edge)}
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(base_model_id, **processor_kwargs)

    # Load base VLM
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        torch_dtype=_maybe_bf16(),
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    )

    # Try to apply PEFT adapter; if inference artifact contains no adapter, keep base model
    try:
        model = PeftModel.from_pretrained(model, model_dir)
    except Exception:
        pass

    model.to(DEVICE)
    model.eval()

    return {"model": model, "processor": processor, "base_model_id": base_model_id}

def input_fn(request_body: str, content_type: str = "application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")
    return json.loads(request_body)

def predict_fn(data: Dict[str, Any], mdl: Dict[str, Any]):
    model = mdl["model"]
    processor = mdl["processor"]

    image = _load_image_from_payload(data)
    prompt = data.get("prompt", "Give a concise caption.")
    max_new_tokens = int(data.get("max_new_tokens", 64))
    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 0.9))

    # SmolVLM chat template
    messages = _build_messages(prompt, num_images=1)
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # The decoded string typically starts with "Assistant: " in chat-style models
    text = outputs[0]
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()

    return {"generated_text": text}

def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    return json.dumps(prediction), accept
