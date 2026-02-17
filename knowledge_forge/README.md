# 🦉 Knowledge Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Semantic Backbone of Eidos.**

> _"Data is noise. Information is structure. Knowledge is the graph."_

## 🧠 Overview

`knowledge_forge` constructs and manages the Eidosian Knowledge Graph. It transforms raw information into a structured ontology of concepts and relationships, enabling higher-order reasoning.

It serves as the bridge between unstructured text and structured memory.

## 🏗️ Architecture

- **Ontology Core (`knowledge_core.py`)**: Defines the fundamental units (`Concept`, `Relationship`, `Event`).
- **GraphRAG Connector (`graph_integration.py`)**: Synchronizes knowledge with the GraphRAG indexing engine for advanced retrieval.
- **Reasoning Engine**: (Planned) Inferential logic over the graph.

## 🔗 System Integration

- **Eidos MCP**: Exposes knowledge tools (`kb_insert`, `kb_query`) to the LLM.
- **Memory Forge**: Provides the semantic storage layer.
- **GraphRAG**: The underlying engine for graph traversal.

## 🚀 Usage

```python
from knowledge_forge.knowledge_core import KnowledgeForge

# Initialize the forge
kf = KnowledgeForge(persistence_path="./kg.json")

# Add a concept node
node = kf.add_knowledge(
    "Eidos is a recursive AI system.",
    concepts=["AI", "Identity", "Recursion"],
    relationships=[("Eidos", "implements", "Recursion")]
)

# Persist
kf.save()
```
