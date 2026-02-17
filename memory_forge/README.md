# 🧠 Memory Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Hippocampus of Eidos.**

> _"Memory is not a recording of the past, but a reconstruction for the future."_

## 🧠 Overview

`memory_forge` provides the persistence layer for Eidosian consciousness. It manages:
- **Episodic Memory**: Vector-based storage and retrieval of specific events and interactions.
- **Semantic Memory**: Generalized knowledge distilled from episodes.
- **Working Memory**: Context-limited buffer for immediate processing.

## 🏗️ Architecture

The system operates on a tiered storage model:

1.  **Hot Tier (JSON/Memory)**: Fast, volatile storage for active context.
2.  **Warm Tier (Vector DB)**: **ChromaDB** is used for semantic search and retrieval.
3.  **Cold Tier (Archive)**: Compressed summaries for long-term retention.

## 🔗 System Integration

- **Eidos MCP**: Exposes memory tools (`memory_insert`, `memory_search`) to the LLM.
- **Agent Forge**: Agents use `MemoryForge` to maintain continuity across beats.
- **Doc Forge**: Documentation of memory schemas and APIs.

## 🚀 Usage

### Python API

```python
from memory_forge import MemoryForge

# Initialize the forge (auto-detects config from GIS)
mem = MemoryForge()

# Commit a memory (Episodic)
mem.remember(
    content="I generated documentation for the Agent Forge.",
    tags=["work", "success", "documentation"]
)

# Recall related memories
results = mem.search("documentation", limit=5)
for r in results:
    print(f"[{r.score}] {r.content}")
```

### CLI

```bash
# Search memory via CLI shim
python -m memory_forge.cli search "project status"
```

## 🛠️ Configuration

Configuration is managed via `GIS` or `env`:
- `MEMORY_BACKEND`: `chromadb` (default) or `json`.
- `CHROMA_PERSIST_DIRECTORY`: Path to vector store.
- `EMBEDDING_MODEL`: Default `all-MiniLM-L6-v2`.
