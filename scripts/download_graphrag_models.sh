#!/bin/bash
set -euo pipefail

mkdir -p models

echo "Downloading Llama 3.2 1B Instruct..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir models --local-dir-use-symlinks False

echo "Downloading Qwen 2.5 0.5B Instruct..."
huggingface-cli download bartowski/Qwen2.5-0.5B-Instruct-GGUF Qwen2.5-0.5B-Instruct-Q8_0.gguf --local-dir models --local-dir-use-symlinks False

echo "Downloading Nomic Embed Text..."
huggingface-cli download lmstudio-community/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False

echo "Model downloads complete."
ls -lh models/
