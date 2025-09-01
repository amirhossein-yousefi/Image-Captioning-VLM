import argparse
import os
import sagemaker
from sagemaker.huggingface import HuggingFace

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}

def main():
    parser = argparse.ArgumentParser()
    # infra
    parser.add_argument("--role", type=str, default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--instance_type", type=str, default="ml.g5.2xlarge")
    parser.add_argument("--instance_count", type=int, default=1)
    parser.add_argument("--volume_size", type=int, default=200)
    parser.add_argument("--max_runtime_seconds", type=int, default=3 * 60 * 60)
    parser.add_argument("--spot", type=str2bool, default=False)
    parser.add_argument("--transformers_version", type=str, default="4.44")
    parser.add_argument("--pytorch_version", type=str, default="2.4")
    parser.add_argument("--py_version", type=str, default="py310")
    parser.add_argument("--output_path", type=str, default=None)  # s3:// bucket/prefix for model.tar.gz

    # these mirror your README/run-args
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--image_longest_edge", type=int, default=1536)
    parser.add_argument("--use_qlora", type=str2bool, default=True)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if not args.role:
        raise SystemExit("Must provide --role (an IAM role with SageMaker permissions).")

    sm_session = sagemaker.Session()
    # Your training script sits in repo root; Estimator will tar & upload source_dir to the container.
    estimator = HuggingFace(
        role=args.role,
        py_version=args.py_version,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        entry_point="train_vlm_sft.py",   # <— uses the same CLI your README documents
        source_dir=".",                   # <— repo root
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        max_run=args.max_runtime_seconds,
        sagemaker_session=sm_session,
        output_path=args.output_path,     # optional S3 prefix to store model artifact
        hyperparameters={
            # IMPORTANT: save to SM_MODEL_DIR (/opt/ml/model) so SageMaker uploads the artifact for hosting
            "output_dir": "/opt/ml/model",
            "base_model_id": args.base_model_id,
            "dataset_id": args.dataset_id,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "image_longest_edge": args.image_longest_edge,
            "use_qlora": "true" if args.use_qlora else "false",
        },
        environment={
            # Faster HF Hub pulls in training jobs
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Some images rely on this to detect #GPUs available
            "SM_NUM_GPUS": os.environ.get("SM_NUM_GPUS", ""),
        },
        keep_alive_period_in_seconds=60,
        enable_sagemaker_metrics=True,
        use_spot_instances=args.spot,
        max_wait=args.max_runtime_seconds if args.spot else None,
    )

    # Launch one-click job. If you have S3 datasets, pass {"train": "s3://..."} as fit() channels.
    estimator.fit(wait=True)

    print("\n=== SageMaker training job completed ===")
    print("S3 model artifact:", estimator.model_data)
    print("Copy this into deploy_sm_endpoint.py via --model_data")

if __name__ == "__main__":
    main()
