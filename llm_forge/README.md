# LLM Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Interface to Synthetic Intelligence.**

## 🤖 Overview

`llm_forge` provides a unified interface for interacting with various Large Language Models.
It abstracts away provider differences (OpenAI vs. Local/HuggingFace) and provides tools for model comparison and evaluation.

## 🏗️ Architecture
- `src/llm_forge/`: Core logic.
    - `model_manager.py`: Handles loading and context.
    - `comparison_generator.py`: Runs side-by-side prompts.
    - `trainer.py`: Fine-tuning utilities.

## 🚀 Usage

```bash
# Compare two models
llm-forge compare --prompt "What is the meaning of life?" --models "gpt-4,local-llama"
```