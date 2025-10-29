import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/final_model")
    parser.add_argument("--repo_id", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN no encontrado. Crea un token con permiso write en https://huggingface.co/settings/tokens y exporta HF_TOKEN antes de ejecutar.")

    if not Path(args.model_dir).exists():
        raise SystemExit(f"Directorio de modelo no existe: {args.model_dir}")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)

    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        repo_type="model",
        token=token,
    )

    print(f"Subido a Hugging Face Hub: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
