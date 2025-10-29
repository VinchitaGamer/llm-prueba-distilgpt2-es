import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space_repo", required=True)
    parser.add_argument("--model_repo", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN no encontrado. Crea un token con permiso write y exporta HF_TOKEN antes de ejecutar.")

    api = HfApi(token=token)
    api.create_repo(args.space_repo, repo_type="space", exist_ok=True, space_sdk="gradio")

    deploy_dir = Path(__file__).resolve().parent
    upload_folder(repo_id=args.space_repo, folder_path=str(deploy_dir), repo_type="space", token=token)

    url = f"https://huggingface.co/spaces/{args.space_repo}"
    print(f"Space desplegado: {url}")
    print("En Settings del Space, define la variable MODEL_ID con el repo del modelo, por ejemplo:", args.model_repo)


if __name__ == "__main__":
    main()
