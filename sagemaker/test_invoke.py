import argparse
import base64
import json
import os
import boto3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint_name", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image_url", type=str)
    g.add_argument("--image_path", type=str)
    p.add_argument("--prompt", type=str, default="Give a concise caption.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    args = p.parse_args()

    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.image_url:
        payload["image_url"] = args.image_url
    else:
        with open(args.image_path, "rb") as f:
            payload["image"] = base64.b64encode(f.read()).decode("utf-8")

    smrt = boto3.client("sagemaker-runtime")
    resp = smrt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    print(resp["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()
