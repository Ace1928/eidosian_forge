#!/bin/bash
# Script to download the Qwen 2.5 1.5B Instruct GGUF model

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null
then
    echo "huggingface-cli could not be found."
    echo "Attempting to install huggingface_hub[cli]..."
    pip install -U "huggingface_hub[cli]"
fi

# Ensure doc_forge/models directory exists
mkdir -p doc_forge/models

# Download the model
# Using q5_k_m for balanced quality and speed
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q5_k_m.gguf --local-dir ./doc_forge/models --local-dir-use-symlinks False
