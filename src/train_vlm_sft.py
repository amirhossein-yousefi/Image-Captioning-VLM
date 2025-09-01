import argparse, os, math, random, sys
from typing import List, Dict, Any
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    # Models: SmolVLM (default) → easy & Apache-2.0; try Idefics3 for Llama3-based VLMs
    p.add_argument("--base_model_id", type=str,
                   default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--dataset_id", type=str,
                   default="jxie/coco_captions")  # image+caption
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--output_dir", type=str, default="outputs/smolvlm-coco-lora")
    p.add_argument("--max_samples", type=int, default=0, help="0=all")
    p.add_argument("--image_longest_edge", type=int, default=1536,
                   help="Multiple of 384 is best for SmolVLM")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=2)
    p.add_argument("--lora_alpha", type=int, default=2)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--freeze_vision", action="store_true", default=True)
    p.add_argument("--tune_projector", action="store_true",
                   help="Unfreeze mm projector in addition to LoRA on text")
    return p.parse_args()

class VLMCaptionCollator:
    """
    Builds batches:
      - user-only chat prompt with image ("Describe this image in detail.")
      - labels only on the assistant target (caption + </s>)
      - pads variable-length sequences; stacks pixel_values
    """
    def __init__(self, processor: AutoProcessor, max_seq_len: int = 1024):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_seq_len = max_seq_len

    def _build_prompt(self) -> List[Dict[str, Any]]:
        return [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text",
                              "text": "Describe this image in detail."}]}]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in examples]
        targets = [ex["caption"] for ex in examples]
        # 1) User-only prompts
        prompts = [self.processor.apply_chat_template(
                       [self._build_prompt()[0]], add_generation_prompt=True)
                   for _ in targets]
        # 2) Process user prompts + images to tensors
        proc = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
            size={"longest_edge": self.processor.image_processor.size.get("longest_edge", 1536)},
        )
        input_ids_list, attn_list, label_list = [], [], []
        for i, tgt in enumerate(targets):
            prompt_ids = proc["input_ids"][i]
            tgt_ids = self.tok(
                tgt + self.tok.eos_token,
                add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            # concat prompt + target
            ids = torch.cat([prompt_ids, tgt_ids], dim=0)
            labels = torch.cat([torch.full_like(prompt_ids, -100), tgt_ids], dim=0)
            attn = torch.ones_like(ids)
            # truncate from the left if too long
            if ids.size(0) > self.max_seq_len:
                cut = ids.size(0) - self.max_seq_len
                ids = ids[cut:]
                labels = labels[cut:]
                attn = attn[cut:]
            input_ids_list.append(ids)
            attn_list.append(attn)
            label_list.append(labels)
        # 3) pad with tokenizer
        batch = self.tok.pad(
            {"input_ids": input_ids_list,
             "attention_mask": attn_list,
             "labels": label_list},
            padding=True,
            return_tensors="pt"
        )
        batch["pixel_values"] = proc["pixel_values"]
        return batch

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset (COCO captions) ---
    ds = load_dataset(args.dataset_id)
    train = ds[args.train_split]
    evald = ds[args.eval_split] if args.eval_split in ds else None
    test=ds['test']
    train=train.select(range(2000))
    evald=evald.select(range(500))
    test=test.select(range(500))



    def ensure(example):
        # jxie/coco_captions provides 'image' + 'caption'
        return {"image": example["image"], "caption": example["caption"]}
    train = train.map(ensure, remove_columns=[c for c in train.column_names if c not in ["image", "caption"]])
    if evald:
        evald = evald.map(ensure, remove_columns=[c for c in evald.column_names if c not in ["image", "caption"]])

    if args.max_samples and args.max_samples > 0:
        train = train.select(range(min(args.max_samples, len(train))))
        if evald: evald = evald.select(range(min(max(args.max_samples // 10, 200), len(evald))))

    # --- Processor & model ---
    processor = AutoProcessor.from_pretrained(args.base_model_id)
    # align processor image size for SmolVLM (multiples of 384 recommended)
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": args.image_longest_edge}

    quant_config = None
    if args.use_qlora and device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16 if args.bf16 and device == "cuda" else "auto",
        # attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        quantization_config=quant_config,
        device_map="auto" if quant_config is not None else None,
    )

    # Freeze vision tower if requested
    if args.freeze_vision:
        for name, param in model.named_parameters():
            if "vision" in name or "image" in name:
                param.requires_grad = False

    # LoRA on text decoder + (optionally) keep projector trainable
    if args.use_lora:
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            modules_to_save=["multi_modal_projector"] if args.tune_projector else None,
        )
        model = get_peft_model(model, lora)

    # --- Trainer ---
    collator = VLMCaptionCollator(processor, max_seq_len=args.max_seq_len)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps" if evald else "no",
        eval_steps=50,
        bf16=args.bf16 and device == "cuda",
        fp16=not args.bf16 and device == "cuda",
        gradient_checkpointing=True,
        report_to=['tensorboard'],
        remove_unused_columns=False,
        logging_dir=os.path.join(args.output_dir,'logs')
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train,
        eval_dataset=evald,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Simple summary
    n = len(train)
    total_steps = math.ceil((n * args.epochs) / (args.batch_size * args.grad_accum))
    print(f"Done. ~{total_steps} steps. Saved to: {args.output_dir}")
    test_metric=trainer.evaluate(test)
    print(test_metric)
if __name__ == "__main__":
    main()
