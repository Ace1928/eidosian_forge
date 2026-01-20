# Knowledge Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Semantic Backbone of Eidos.**

## 🧠 Overview

`knowledge_forge` constructs and manages the Eidosian Knowledge Graph. It provides tools for:
- **Ontology Management**: Defining concepts and relationships.
- **Graph Reasoning**: Inferring new knowledge from existing data.
- **Integration**: Connecting with `graphrag` for retrieval.

## 🏗️ Architecture
- `knowledge_core.py`: Basic node/edge implementation.
- `graph_integration.py`: Connectors to external systems.

## 🚀 Usage

```python
from knowledge_forge.knowledge_core import KnowledgeForge

kf = KnowledgeForge(persistence_path="./kg.json")
node = kf.add_knowledge("Eidos is an AI", concepts=["AI", "Identity"])
kf.save()
```