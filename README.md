# Open VLM fine-tuning (SmolVLM + COCO captions)

## 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (If on NVIDIA GPU & want QLoRA) ensure bitsandbytes is available; otherwise run with --use_qlora false
```



## 1)Fine-tune (LoRA/QLoRA, frozen vision tower)
```bash
python train_vlm_sft.py \
  --base_model_id HuggingFaceTB/SmolVLM-Instruct \
  --dataset_id jxie/coco_captions \
  --output_dir outputs/smolvlm-coco-lora \
  --epochs 1 --batch_size 2 --grad_accum 8 \
  --max_seq_len 1024 --image_longest_edge 1536

```

## 2) Inference
```bash
python inference_vlm.py \
  --base_model_id HuggingFaceTB/SmolVLM-Instruct \
  --adapter_dir outputs/smolvlm-coco-lora \
  --image https://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg \
  --prompt "Give a concise caption."
```

## 3) Evaluation
```bash
python eval_caption_metric.py
```