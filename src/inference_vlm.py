import argparse, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--adapter_dir", default="outputs/smolvlm-coco-lora")
    p.add_argument("--image", required=True, help="Path or URL")
    p.add_argument("--prompt", default="Describe this image in detail.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()

def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.adapter_dir)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model_id, torch_dtype=torch.bfloat16 if device=="cuda" else "auto"
    ).to(device)
    # If you trained with LoRA, the adapter weights are inside adapter_dir;
    # AutoModelForVision2Seq.from_pretrained(adapter_dir) also works if you saved merged weights there.
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    except Exception:
        pass

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type":"image"}, {"type":"text","text": args.prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(out)

if __name__ == "__main__":
    main()
