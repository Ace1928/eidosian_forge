# LLM Forge Installation Guide

## Quick Install

```bash
# From the llm_forge directory
pip install -e .

# Or install with OpenAI support
pip install -e ".[openai]"
```

## Verify Installation

```bash
# Check CLI is available
llm-forge --version

# Check system status
llm-forge status
```

## Prerequisites

LLM Forge uses Ollama as the default backend. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull phi3:mini

# Pull embedding model
ollama pull nomic-embed-text
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/llm_forge/completions/llm-forge.bash

# Or for a one-time session
source completions/llm-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
LLM Forge works independently with these capabilities:
- Chat with local Ollama models
- Generate text embeddings
- List and manage models
- Response caching

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `memory_forge` | Conversational memory |
| `knowledge_forge` | RAG-enhanced responses |

Check available integrations:
```bash
llm-forge integrations
```

## CLI Commands

```bash
llm-forge status              # Show LLM status
llm-forge models              # List available models
llm-forge chat <prompt>       # Chat with LLM
llm-forge embed <text>        # Generate embeddings
llm-forge config              # Show model configuration
llm-forge test                # Test LLM connection
```

## Python API

```python
from llm_forge import ModelManager, OllamaProvider

# Using Model Manager
manager = ModelManager()
response = manager.ollama_provider.generate(
    prompt="Explain quantum computing",
    model="phi3:mini",
    temperature=0.7,
)
print(response.content)

# Direct Ollama usage
ollama = OllamaProvider()
response = ollama.generate("Hello, world!")
print(response.content)

# Embeddings
import requests
resp = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "nomic-embed-text", "prompt": "Hello"}
)
embedding = resp.json()["embedding"]
```

## Model Configuration

The unified model configuration is stored at:
```
/home/lloyd/eidosian_forge/data/model_config.json
```

Default models:
- **Inference**: phi3:mini (2.2GB, best reasoning for size)
- **Embedding**: nomic-embed-text (768 dimensions)

## Troubleshooting

### Ollama Not Responding
```bash
# Check if Ollama is running
systemctl status ollama

# Start Ollama
systemctl start ollama

# Or run manually
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull required model
ollama pull phi3:mini
```
