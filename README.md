
# 📸 Image Captioning with SmolVLM — Fine‑Tuning, Inference & Evaluation

> Minimal, practical pipeline to fine‑tune an open **Vision‑Language Model (VLM)** for **image captioning** using LoRA/QLoRA on the **COCO Captions** dataset, then run inference and evaluate — all with a lightweight PyTorch + 🤗 Transformers stack.

<p align="center">
  <img alt="SmolVLM" src="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/smolvlm_card.png" width="520">
  <br>
  <em>Powered by <a href="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct">SmolVLM‑Instruct</a> — an efficient, Apache‑2.0 licensed VLM designed for on‑device use.</em>
</p>

---

## ✨ Highlights

- **Open weights** base: `HuggingFaceTB/SmolVLM-Instruct` (compact multimodal model; Apache‑2.0).
- **LoRA / QLoRA fine‑tuning** with the **vision tower frozen** by default for efficiency.
- **Straightforward training loop** (no heavy frameworks): just `torch`, `transformers`, `datasets`, `peft`, `accelerate`.
- **COCO Captions** out of the box via the 🤗 Datasets Hub (`jxie/coco_captions`).
- **Evaluation script** for caption metrics + a simple **inference script** for images or URLs.
- **LoRA merge** utility to bake adapters into base weights for a single deployable checkpoint.

> If you want to reproduce the exact dependencies/versions used here, see the pinned `requirements.txt` in this repo.

---

## 🚀 Model on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-VLM--Image--Captioning-yellow.svg)](https://huggingface.co/Amirhossein75/VLM-Image-Captioning)
<p align="center">
  <a href="https://huggingface.co/Amirhossein75/VLM-Image-Captioning">
    <img src="https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-blueviolet?style=for-the-badge" alt="Hugging Face Repo">
  </a>
</p>

---
## 📁 Project Structure

```text
Image-Captioning-VLM/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── train_vlm_sft.py          # Supervised fine-tuning w/ LoRA or QLoRA (vision tower frozen)
    ├── inference_vlm.py          # Generate captions for an image (file path or URL)
    ├── eval_caption_metric.py    # Compute caption metrics for predictions
    ├── merge_lora.py             # Merge LoRA adapter into base model weights
    ├── metrics.txt               # (example) saved metrics/logs
    └── outputs/
        └── smolvlm-coco-lora/
            └── logs/             # Example TensorBoard event logs
```

> ℹ️ The repository currently ships only Python code (100%). No license file is provided yet; see [License](#-license).

---

## 🚀 Quickstart

### 0) Environment

```bash
python -m venv .venv
source .venv/bin/activate           # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notes**
- `bitsandbytes` is **optional** and NVIDIA‑only. If you cannot install it (CPU/Apple Silicon), run with `--use_qlora false` (see below).
- A single 8–12 GB NVIDIA GPU is typically sufficient. SmolVLM’s base inference footprint is around ~5 GB; QLoRA can reduce training memory further.

### 1) Fine‑tune on COCO Captions (LoRA/QLoRA, frozen vision tower)

```bash
cd src

python train_vlm_sft.py       --base_model_id HuggingFaceTB/SmolVLM-Instruct       --dataset_id jxie/coco_captions       --output_dir outputs/smolvlm-coco-lora       --epochs 1 --batch_size 2 --grad_accum 8       --max_seq_len 1024 --image_longest_edge 1536
```

**What the flags mean (common ones):**

| Flag | Purpose |
|------|---------|
| `--base_model_id` | HF repo id of the base VLM checkpoint (defaults used in examples). |
| `--dataset_id` | HF Datasets id to train on (e.g., `jxie/coco_captions`). |
| `--output_dir` | Where to write checkpoints, adapters, and logs. |
| `--epochs` | Number of training epochs. |
| `--batch_size` | Per‑device effective batch size (before grad‑accum). |
| `--grad_accum` | Gradient accumulation steps. |
| `--max_seq_len` | Max total tokens (text + image tokens) for SFT. |
| `--image_longest_edge` | Input image resize: longest edge in pixels (1536 ≈ 4×384 — a good default for SmolVLM). |
| `--use_qlora` | (Optional) Enable 4‑bit QLoRA (requires `bitsandbytes`). |

> 💡 Tip: If VRAM is tight, reduce `--image_longest_edge` (e.g., 1152 or 768) and/or enable QLoRA.

### 2) Inference (generate a caption for a single image)

```bash
python inference_vlm.py       --base_model_id HuggingFaceTB/SmolVLM-Instruct       --adapter_dir outputs/smolvlm-coco-lora       --image https://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg       --prompt "Give a concise caption."
````

**Examples**
- Local file: `--image ./samples/dog.jpg`
- Remote URL: `--image https://.../photo.jpg`
- Plain captioning: `--prompt "Describe the image in one sentence."`

> If `--adapter_dir` is omitted, inference runs on the base SmolVLM‑Instruct weights.

### 3) Evaluation (caption metrics)

```bash
python eval_caption_metric.py
```

The script computes common captioning metrics (e.g., BLEU/METEOR/CIDEr as implemented) for your predictions. Check `src/metrics.txt` and the console output for exact metrics produced.

---

## 🧰 LoRA → Full Weights (Merging)

After you’re happy with the adapter, you can merge it into the base model to get a single checkpoint for deployment:

```bash
python merge_lora.py       --base_model_id HuggingFaceTB/SmolVLM-Instruct       --adapter_dir outputs/smolvlm-coco-lora       --save_dir outputs/smolvlm-coco-merged
```

The merged model in `--save_dir` can be pushed to the Hub or loaded with standard `AutoModelForVision2Seq` APIs.

---

## 📚 Dataset

**COCO Captions (Hugging Face Hub: `jxie/coco_captions`)**

- ~**616k** caption rows across train/val/test splits.
- Download size ≈ **20.9 GB** (auto‑converted parquet ~20.9 GB).
- Each image has **5 human captions** (classic COCO captioning setup).

> If you are on limited bandwidth/disk, try a smaller split, subsample, or an alternate dataset (e.g., Flickr8k/30k).

---

## 🔧 Implementation Notes

- **Model family**: SmolVLM is a compact VLM using **SigLIP** as the image encoder + **SmolLM2** as the language decoder. It compresses visuals to **81 tokens per 384×384 patch**, enabling fast, memory‑efficient training/inference. The default image size used here (`--image_longest_edge 1536`) corresponds to 4× the 384 px patch size — a robust general‑purpose setting.
- **Precision & memory**: For inference, BF16/FP16 is recommended. For training on small GPUs, **QLoRA** (4‑bit) is supported via `bitsandbytes`. You can also downscale images (`--image_longest_edge`) to fit tighter VRAM budgets.
- **Frameworks**: Pure PyTorch + 🤗 Transformers/Datasets + PEFT adapters + Accelerate launcher under the hood.

---

## 📈 Tips for Better Captions

- Preprocess images to the target `--image_longest_edge` to avoid runtime resizing overhead.
- Use **grad‑accumulation** to simulate larger batch sizes on limited VRAM.
- Start with **1–2 epochs**; overfitting comes quick on COCO when the vision tower is frozen.
- Try curriculum prompts during SFT (e.g., “Describe the **main** objects” → “Describe the **scene**”).

---

## 🧪 Reproducibility & Hardware

- Typical inference VRAM for SmolVLM‑Instruct is ~**5 GB** (varies by attention backend/precision).
- Training with LoRA + frozen vision tower runs comfortably on **8–12 GB** GPUs; use QLoRA if needed.
- Logs are written under `outputs/<run>/logs` (TensorBoard event files). Use `tensorboard --logdir outputs`.

---

## 🧩 Troubleshooting

- **`bitsandbytes` fails to import** → install CUDA‑aligned wheels or run with `--use_qlora false`.
- **CUDA OOM** → reduce `--image_longest_edge`, increase `--grad_accum`, or enable QLoRA.
- **Tokenizer pad/truncation warnings** → lower `--max_seq_len` or ensure conversations fit the window.
- **HF Hub rate limits** → set `HF_HUB_ENABLE_HF_TRANSFER=1` and use a token if needed.

---

## 🤝 Contributing

Pull requests are welcome! Good first contributions:

- Add new datasets (Flickr8k/30k, VizWiz, TextCaps, etc.).
- Add mixed‑precision flags & Flash‑Attention toggle.
- Expand `eval_caption_metric.py` with more metrics and HTML report.
- CI for style checks (ruff/black) and smoke tests.

---

## 📜 License

This repository **currently does not include a license file**. Until a license is added by the author, contributions/usage are governed by default “all rights reserved”. If you intend to open‑source this, consider adding **Apache‑2.0** (compatible with SmolVLM’s license), **MIT**, or another permissive license.

---

## 🙌 Acknowledgements & References

- **Base model:** [HuggingFaceTB/SmolVLM‑Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) (Apache‑2.0).
- **Dataset:** [jxie/coco_captions](https://huggingface.co/datasets/jxie/coco_captions).
- **SmolVLM paper:** Marafioti et al., 2025. *SmolVLM: Redefining small and efficient multimodal models*.

---

## 🔗 Citation

If this repo helps your research, please consider citing the SmolVLM technical report and this repository.

```bibtex
@article{marafioti2025smolvlm,
  title={SmolVLM: Redefining small and efficient multimodal models},
  author={Marafioti, Andr{\'e}s and Zohar, Orr and Farr{\'e}, Miquel and Noyan, Merve and Bakouch, Elie and Cuenca, Pedro and Zakka, Cyril and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and Srivastav, Vaibhav and Lochner, Joshua and Larcher, Hugo and Morlon, Mathieu and Tunstall, Lewis and von Werra, Leandro and Wolf, Thomas},
  journal={arXiv preprint arXiv:2504.05299},
  year={2025}
}
```
