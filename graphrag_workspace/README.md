# 🕸️ GraphRAG Workspace

**The Semantic Web of Eidos.**

> _"From text to topology."_

## 🕸️ Overview

This directory contains the operational configuration and workspace for **Microsoft GraphRAG**, adapted for the Eidosian local environment. It is designed to ingest textual data from `input/`, process it using local LLMs, and generate a traversable knowledge graph.

## ⚙️ Configuration (`settings.yaml`)

The system is configured to use **local inference** endpoints to maintain data sovereignty and zero cost.

- **Chat Model**: `phi3:mini` (via Ollama)
- **Embedding Model**: `nomic-embed-text` (via Ollama)
- **Endpoint**: `http://localhost:11434/v1`

### Requirements
1.  **Ollama**: Must be running (`ollama serve`).
2.  **Models**: Pull the required models:
    ```bash
    ollama pull phi3:mini
    ollama pull nomic-embed-text
    ```

## 🚀 Usage

### Indexing
Run the indexing pipeline to build the graph:

```bash
python -m graphrag.index --root graphrag_workspace
```

### Querying
**Local Search** (Specific details):
```bash
python -m graphrag.query --root graphrag_workspace --method local "Who is Annastasia?"
```

**Global Search** (Thematic summaries):
```bash
python -m graphrag.query --root graphrag_workspace --method global "What are the core themes of the emails?"
```

## 🔗 Integration
This workspace is managed by `knowledge_forge`. The `grag_index` and `grag_query` tools in the Nexus map to these commands.
