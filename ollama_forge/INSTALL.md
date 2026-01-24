# 🦙 Ollama Forge Installation

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

## Quick Install

```bash
# Clone (if standalone)
git clone https://github.com/Ace1928/ollama_forge.git
cd ollama_forge

# Install
pip install -e .

# Or as part of Eidosian Forge
pip install -e ./ollama_forge
```

## Verify Installation

```bash
# Ensure Ollama is running
ollama serve

# Test the forge
python -c "from ollama_forge import OllamaClient; print('✓ Ready')"
```

## CLI Usage

```bash
# List available models
ollama-forge list

# Generate text
ollama-forge generate "Hello, world!" --model qwen2.5:1.5b

# Help
ollama-forge --help
```

## Configuration

The forge connects to Ollama at `http://localhost:11434` by default.

Set `OLLAMA_HOST` environment variable to use a different endpoint:
```bash
export OLLAMA_HOST=http://192.168.1.100:11434
```

## Dependencies

- `requests` - HTTP client
- `eidosian_core` - Universal decorators and logging

