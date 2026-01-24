# Memory Forge Installation Guide

## Quick Install

```bash
# From the memory_forge directory
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check CLI is available
memory-forge --version

# Check system status
memory-forge status
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/memory_forge/completions/memory-forge.bash

# Or for a one-time session
source completions/memory-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
Memory Forge works independently with these capabilities:
- Store and retrieve tiered memories
- Automatic memory introspection
- Context-based memory suggestions
- Memory cleanup and maintenance

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `knowledge_forge` | Unified memory + knowledge search |
| `llm_forge` | Smart memory summarization |
| `word_forge` | Semantic relationship mapping |

Check available integrations:
```bash
memory-forge integrations
```

## Configuration

Memory data is stored in:
```
/home/lloyd/eidosian_forge/data/memory/
├── short_term.json
├── working.json
├── long_term.json
├── self.json
└── user.json
```

## CLI Commands

```bash
memory-forge status          # Check system health
memory-forge list            # List memories
memory-forge search <query>  # Search memories
memory-forge store <text>    # Store new memory
memory-forge introspect      # Analyze patterns
memory-forge context <prompt># Get auto-context
memory-forge stats           # View statistics
memory-forge cleanup         # Remove expired
```

## Python API

```python
from memory_forge import TieredMemorySystem, MemoryTier, MemoryNamespace

# Initialize
memory = TieredMemorySystem()

# Store a memory
memory_id = memory.remember(
    content="Important information",
    tier=MemoryTier.LONG_TERM,
    namespace=MemoryNamespace.KNOWLEDGE,
    tags={"important", "reference"},
)

# Search memories
results = memory.recall("important", limit=5)

# Get auto-context
from memory_forge import AutoContextEngine
context = AutoContextEngine(memory)
suggestions = context.suggest_context("What do I know about X?")
```

## Troubleshooting

### Import Errors
Ensure the lib directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### Permission Issues
Ensure the data directory is writable:
```bash
chmod 755 /home/lloyd/eidosian_forge/data/memory
```
