import os
from huggingface_hub import (
    snapshot_download,
    HfApi,
    login,
    ModelCard,
)
from typing import Optional


def download_model(
    model_id: str, local_root: str = "~/Development/models"
) -> Optional[str]:
    """Download a model from Hugging Face Hub with full repository contents"""
    try:
        # Extract group name from model ID
        group_name = model_id.split("/")[0] if "/" in model_id else "misc"
        model_name = model_id.split("/")[-1]

        # Create local directory structure
        local_path = os.path.expanduser(
            os.path.join(local_root, group_name, model_name)
        )
        os.makedirs(local_path, exist_ok=True)

        # Download all repository files
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            allow_patterns=[
                "*.json",
                "*.bin",
                "*.model",
                "*.py",
                "*.md",
                "*.txt",
                "*.safetensors",
            ],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
        )

        # Get and save model card
        card = ModelCard.load(model_id)
        with open(os.path.join(local_path, "README.md"), "w") as f:
            f.write(str(card.content))

        print(f"\nSuccessfully downloaded {model_id} to {local_path}")
        return path

    except Exception as e:
        print(f"\nError downloading {model_id}: {str(e)}")
        return None


def find_similar_models(model_id: str, top_k: int = 5) -> list:
    """Search for similar models using Hugging Face API"""
    api = HfApi()
    query = model_id.split("/")[-1]  # Use model name for search

    # Search models with similar names
    models = api.list_models(filter=query, sort="downloads", direction=-1, limit=top_k)

    return [model_id for model_id in models if model_id != model_id]


def main():
    while True:
        model_id = input(
            "\nEnter Hugging Face model ID (e.g. 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B') or 'q' to quit: "
        ).strip()

        if model_id.lower() == "q":
            break

        result = download_model(model_id)

        if not result:
            print(f"\nModel '{model_id}' not found. Searching for similar models...")
            similar = find_similar_models(model_id)

            if similar:
                print("\nDid you mean one of these?")
                for i, name in enumerate(similar, 1):
                    print(f"{i}. {name}")

                choice = input(
                    "\nEnter number to download (or 'n' to try again): "
                ).strip()
                if choice.isdigit() and 1 <= int(choice) <= len(similar):
                    download_model(similar[int(choice) - 1])
            else:
                print("No similar models found. Please check the model ID.")


if __name__ == "__main__":
    main()
