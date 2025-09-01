import argparse, torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base_model_id", default="HuggingFaceTB/SmolVLM-Instruct")
p.add_argument("--adapter_dir", required=True)
p.add_argument("--out_dir", default="outputs/merged-sft")
args = p.parse_args()

processor = AutoProcessor.from_pretrained(args.adapter_dir)
base = AutoModelForVision2Seq.from_pretrained(args.base_model_id, torch_dtype="auto")
merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
merged.save_pretrained(args.out_dir)
processor.save_pretrained(args.out_dir)
print(f"Merged model saved to: {args.out_dir}")
