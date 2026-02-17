# 🦙 Ollama Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Local Intelligence Interface.**

> _"Silicon minds running on local silicon."_

## 🧠 Overview

`ollama_forge` provides a robust, Pythonic interface to local LLMs via [Ollama](https://ollama.com/). It abstracts the API details, providing async support, error handling, and model management.

## 🏗️ Architecture

- **Client (`OllamaClient`)**: Unified interface for generation and embeddings.
- **Model Manager**: Tools to pull, delete, and inspect local models.
- **Resilience**: Automatic retries and fallback handling.

## 🔗 System Integration

- **Eidos MCP**: Uses `ollama_forge` for local inference tasks when `llama.cpp` is not available or preferred.
- **Agent Forge**: Agents use this to "think" without external dependencies.

## 🚀 Usage

```python
import asyncio
from ollama_forge import AsyncOllamaClient

async def chat():
    client = AsyncOllamaClient()
    response = await client.generate(
        model="qwen2.5:1.5b",
        prompt="Explain recursion."
    )
    print(response.text)

asyncio.run(chat())
```
