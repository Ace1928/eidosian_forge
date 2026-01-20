# Ollama Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Local Intelligence.**

## 🦙 Overview

`ollama_forge` provides a Pythonic wrapper around the [Ollama](https://ollama.com/) API.
It handles:
- **Client**: Async and Sync clients.
- **Model Management**: Pulling, deleting, and listing models.
- **Generation**: Streaming and batch generation.

## 🏗️ Architecture
- `src/ollama_forge/`: Core client.

## 🚀 Usage

```python
from ollama_forge import OllamaClient

client = OllamaClient()
response = client.generate(model="llama3", prompt="Hi")
print(response.text)
```