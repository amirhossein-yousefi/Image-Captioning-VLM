import argparse
import os
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", type=str, default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    p.add_argument("--model_data", type=str, required=True)  # S3 path to model.tar.gz from training
    p.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--endpoint_name", type=str, default="vlm-captioning")
    p.add_argument("--instance_type", type=str, default="ml.g5.2xlarge")
    p.add_argument("--instance_count", type=int, default=1)
    p.add_argument("--transformers_version", type=str, default="4.44")
    p.add_argument("--pytorch_version", type=str, default="2.4")
    p.add_argument("--py_version", type=str, default="py310")
    p.add_argument("--image_longest_edge", type=int, default=1536)
    args = p.parse_args()

    if not args.role:
        raise SystemExit("Must provide --role or set SAGEMAKER_ROLE_ARN.")

    sm_sess = sagemaker.Session()

    # Use the HF Inference DLC + our custom handler in inference/
    model = HuggingFaceModel(
        role=args.role,
        model_data=args.model_data,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        py_version=args.py_version,
        entry_point="inference.py",
        source_dir="inference",
        sagemaker_session=sm_sess,
        env={
            "BASE_MODEL_ID": args.base_model_id,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "IMAGE_LONGEST_EDGE": str(args.image_longest_edge),
        },
    )

    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )

    print("\n=== Endpoint deployed ===")
    print("Name:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
