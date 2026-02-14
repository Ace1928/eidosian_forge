import sys
import logging
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig
import os


# Setup logging
logging.basicConfig(level=logging.INFO)

# Configuration and Constants
API = HfApi()
CONSOLE = Console()
MODELS_LIMIT = 50  # Limit number of models to retrieve for demo; adjust as needed.

def authenticate_hf_hub(token):
    """
    Authenticate with the Hugging Face Hub using a token.
    """
    HfFolder.save_token(token)

# Example usage (replace 'your_token_here' with your actual token)
authenticate_hf_hub("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")

def fetch_open_source_models(limit=MODELS_LIMIT):
    """
    Fetch models from Hugging Face Hub filtered by a heuristic tag 
    to emphasize those typically supported by transformers.
    """
    # Retrieve models with a specific tag; 'transformers' tag heuristically filters.
    models = API.list_models(filter="transformers", sort="downloads", direction=-1, limit=limit)
    open_source_models = []

    for model in models:
        # Check repository visibility (public implies open source) and try loading config via transformers.
        if model.private:
            continue
        try:
            # Attempt to load model configuration to check compatibility.
            _ = AutoConfig.from_pretrained(model.modelId)
            open_source_models.append(model)
        except Exception as e:
            logging.error(f"Failed to load model configuration for {model.modelId}: {e}")
            continue

    return open_source_models

def display_models(models):
    """
    Display models in a user-friendly table format.
    """
    table = Table(title="Available Transformers Models")
    table.add_column("Index", justify="right")
    table.add_column("Model ID", style="cyan")

    for idx, model in enumerate(models):
        table.add_row(str(idx), model.modelId)

    CONSOLE.print(table)

def select_model(models):
    """
    Let user select a model by index.
    """
    while True:
        try:
            choice = int(input("Select a model by index: "))
            if 0 <= choice < len(models):
                return models[choice]
            else:
                logging.warning(f"Please enter a number between 0 and {len(models)-1}")
        except ValueError as e:
            logging.warning(f"Invalid input: {e}")

def list_available_sizes(model_id):
    """
    List available sizes for a given model. 
    This simplistic approach checks for different file sizes of primary model files.
    """
    try:
        files = API.list_repo_files(repo_id=model_id)
    except Exception as e:
        logging.error(f"Error accessing repository for model {model_id}: {e}")
        return []

    size_info = []
    for filename in files:
        if any(filename.endswith(ext) for ext in [".bin", ".pt", ".ckpt"]):
            try:
                # Use hf_hub_download to get file metadata
                file_path = hf_hub_download(repo_id=model_id, filename=filename)
                size = os.path.getsize(file_path)
                size_info.append((filename, size))
            except Exception as e:
                logging.error(f"Error retrieving file size for {filename}: {e}")
    return size_info

def display_sizes(sizes):
    """
    Display model file sizes in a user-friendly format.
    """
    table = Table(title="Available Model Sizes")
    table.add_column("Index", justify="right")
    table.add_column("Filename", style="magenta")
    table.add_column("Size (bytes)", justify="right")

    for idx, (filename, size) in enumerate(sizes):
        table.add_row(str(idx), filename, str(size))

    CONSOLE.print(table)

def select_size(sizes):
    """
    Let user select a file/size by index.
    """
    while True:
        try:
            choice = int(input("Select a size by index: "))
            if 0 <= choice < len(sizes):
                return sizes[choice]
            else:
                logging.warning(f"Please enter a number between 0 and {len(sizes)-1}")
        except ValueError:
            logging.warning("Invalid input. Please enter a valid integer.")

def list_available_formats(model_id):
    """
    List available formats for the selected model version.
    This simplistic approach treats file extensions as formats.
    """
    try:
        files = API.list_repo_files(repo_id=model_id)
    except Exception as e:
        logging.error(f"Error accessing repository for model {model_id}: {e}")
        return []

    formats = set()
    for filename in files:
        # Derive format from file extension
        ext = filename.split('.')[-1]
        formats.add(ext)

    return list(formats)

def display_formats(formats):
    """
    Display available formats.
    """
    table = Table(title="Available Formats")
    table.add_column("Index", justify="right")
    table.add_column("Format", style="green")

    for idx, fmt in enumerate(formats):
        table.add_row(str(idx), fmt)

    CONSOLE.print(table)

def main():
    # Step 1: List available models
    models = fetch_open_source_models()
    if not models:
        logging.error("No models found.")
        sys.exit(1)
    display_models(models)

    # Step 2: User selects a model
    selected_model = select_model(models)
    logging.info(f"Selected Model: {selected_model.modelId}")

    # Step 3: List available sizes for the selected model
    sizes = list_available_sizes(selected_model.modelId)
    if not sizes:
        print("No size information available.")
    else:
        display_sizes(sizes)
        # Step 4: User selects a size
        selected_size = select_size(sizes)
        logging.info(f"Selected file: {selected_size[0]} of size {selected_size[1]} bytes")

    # Step 5: List available formats for the selected model
    formats = list_available_formats(selected_model.modelId)
    if not formats:
        print("No formats information available.")
    else:
        display_formats(formats)

if __name__ == "__main__":
    main()
