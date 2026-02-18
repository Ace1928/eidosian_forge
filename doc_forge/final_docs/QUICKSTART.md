Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# 🚀 Eidosian Forge Quick Start Guide

> Get up and running with the Eidosian Forge in 5 minutes.

## Prerequisites

- Python 3.12+
- Ollama (for LLM features)

## Installation

### Step-by-step installation:

```bash
cd /home/lloyd/eidosian_forge

# Step 1: Activate virtual environment
source eidosian_venv/bin/activate

# Step 2: Install core forges
pip install -e lib
pip install -e memory_forge
pip install -e knowledge_forge
pip install -e code_forge
pip install -e llm_forge

# Step 3: Enable bash completions (optional)
source bin/eidosian-completion.bash

# Add to ~/.bashrc for permanent completions:
echo 'source /home/lloyd/eidosian_forge/bin/eidosian-completion.bash' >> ~/.bashrc
```

### Quick Verification

```bash
python bin/eidosian status

Expected output:
│      EIDOSIAN FORGE v1.0.0        │
└──────────────────────────────────────╯
 
System Metrics:
   MCP Tools:       92
   Memories:        75
   Knowledge Nodes: 148
   Code Elements:   10498

Forge Status:
   ✓ memory       operational
   ✓ knowledge    operational
   ✓ code         operational
   ✓ llm          operational
```

## Using Individual Forges

### Memory Forge - Tiered Memory System

```bash
memory-forge status

# Store a memory
memory-id = memory.store(
    content="Important meeting at 3pm",
    tier=MemoryTier.LONG_TERM,
    namespace=MemoryNamespace.KNOWLEDGE,
    tags={"meeting"}
)

# Search memories
search_results = memory.search("meeting")

# Get auto-context suggestions
context_result = memory.context("What meetings do I have?")
```

### Knowledge Forge - Knowledge Graph

```bash
knowledge-forge status

list_knowledge_nodes = knowledge-forge.list()

search_knowledge = knowledge-forge.search("architecture")

add_new_knowledge = knowledge-forge.add(
    content="Eidosian Forge uses Python 3.12",
    concepts=["python", "eidos"],
    tags={"important"}
)

find_paths_between_concepts = knowledge-forge.path(node_id, node_id)
```

### Code Forge - Code Analysis

```bash
code-forge status

analyze_code = code-forge.analyze(path/to/file.py)

search_indexed_code = code-forge.search("TieredMemory")

view_stats = code-forge.stats()
```

### LLM Forge - Unified LLM Interface

```bash
llm-forge status

list_available_models = llm-forge.models()

chat_with_model = llm-forge.chat(
    message="Explain the Eidosian philosophy in one sentence"
)

test_connection = llm-forge.test()
```

## Using the Central Hub

The `eidosian` command provides unified access to all forges:

```bash
python bin/eidosian status

# Route to forges
python bin/eidosian memory status
python bin/eidosian knowledge search "architecture"
python bin/eidosian code analyze file.py
python bin/eidosian llm chat "Hello"

# List available forges
python bin/eidosian forges
```

## Python API Quick Examples

### Memory Forge

```python
from memory_forge import TieredMemorySystem, MemoryTier, MemoryNamespace

memory = TieredMemorySystem()

# Store a memory
memory_id = memory.remember(
    content="Important insight about X",
    tier=MemoryTier.LONG_TERM,
    namespace=MemoryNamespace.KNOWLEDGE,
    tags={"insight", "important"}
)

# Recall
results = memory.recall("insight", limit=5)
```

### Knowledge Forge

```python
from knowledge_forge import KnowledgeForge

kb = KnowledgeForge("/home/lloyd/eidosian_forge/data/kb.json")

# Add knowledge
node_id = kb.add_knowledge(
    content="Important concept",
    concepts=["category"],
    tags=["tag1"]
)

# Search
results = kb.search("concept")
```

### Code Forge

```python
from code_forge import CodeIndexer

indexer = CodeIndexer()

# Search code elements
for elem in indexer.elements.values():
    if "memory" in elem.name.lower():
        print(f"{elem.element_type}: {elem.qualified_name}")
```

## MCP Server

Start the MCP server for tool integration:

```bash
cd /home/lloyd/eidosian_forge/eidos_mcp
source ../eidosian_venv/bin/activate
python -m eidos_mcp.eidos_mcp_server
```

## Data Locations

| Location | Path |
|----------|------|
| Memory    | `/home/lloyd/eidosian_forge/data/memory` |
| Knowledge  | `/home/lloyd/eidosian_forge/data/kb.json` |
| Code Index | `/home/lloyd/eidosian_forge/data/code_index.json` |
| Model Config | `/home/lloyd/eidosian_forge/data/model_config.json` |

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### `eidosian_core` Not Found
```bash
cd /home/lloyd/eidosian_forge
source eidosian_venv/bin/activate
pip install -e lib
```

### Ollama Not Running
```bash
ollama serve  # Start Ollama
ollama list   # Check available models
```

### Missing Models
```bash
ollama pull phi3:mini         # Inference model
ollama pull nomic-embed-text  # Embedding model
```

## Next Steps

- Read individual forge INSTALL.md files for detailed documentation
- Explore the MCP tools via `eidos_mcp`
- Check `/home/lloyd/eidosian_forge/docs/FORGE_ARCHITECTURE.md` for system architecture

[ Prompt: 42.0 t/s | Generation: 7.6 t/s ]

Exiting...