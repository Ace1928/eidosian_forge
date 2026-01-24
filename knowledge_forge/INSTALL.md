# Knowledge Forge Installation Guide

## Quick Install

```bash
# From the knowledge_forge directory
pip install -e .

# Or install with all extras
pip install -e ".[dev,viz,rdf]"
```

## Verify Installation

```bash
# Check CLI is available
knowledge-forge --version

# Check system status
knowledge-forge status
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/knowledge_forge/completions/knowledge-forge.bash

# Or for a one-time session
source completions/knowledge-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
Knowledge Forge works independently with these capabilities:
- Store and organize knowledge nodes
- Tag-based and concept-based retrieval
- Bidirectional node linking
- Path finding between concepts
- Full-text search

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `memory_forge` | Unified memory + knowledge search |
| `graphrag` | Advanced semantic indexing |

Check available integrations:
```bash
knowledge-forge integrations
```

## Configuration

Knowledge base is stored at:
```
/home/lloyd/eidosian_forge/data/kb.json
```

## CLI Commands

```bash
knowledge-forge status          # Check graph status
knowledge-forge list            # List nodes
knowledge-forge search <query>  # Search nodes
knowledge-forge add <content>   # Add new node
knowledge-forge link A B        # Link two nodes
knowledge-forge path A B        # Find path
knowledge-forge concepts        # List concepts
knowledge-forge unified <query> # Cross-system search
knowledge-forge stats           # Detailed statistics
knowledge-forge delete <id>     # Remove a node
```

## Python API

```python
from knowledge_forge import KnowledgeForge, KnowledgeNode

# Initialize
kb = KnowledgeForge("/path/to/kb.json")

# Add knowledge
node = kb.add_knowledge(
    content="Important concept",
    concepts=["category1", "category2"],
    tags=["tag1", "tag2"],
)

# Search
results = kb.search("concept")

# Get by concept
nodes = kb.get_by_concept("category1")

# Link nodes
kb.link_nodes(node1_id, node2_id)

# Find path
path = kb.find_path(start_id, end_id)
```

## Unified Search (with memory_forge)

```python
from knowledge_forge import KnowledgeMemoryBridge

bridge = KnowledgeMemoryBridge()
results = bridge.unified_search("important topic", top_k=10)

for r in results:
    print(f"[{r.source}] {r.content[:80]}")
```

## Troubleshooting

### Import Errors
Ensure the lib directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/lloyd/eidosian_forge/lib"
```

### Missing memory_forge for unified search
Install memory_forge for cross-system search:
```bash
cd ../memory_forge && pip install -e .
```
