# 🕸️ GraphRAG Workspace

**The Semantic Web of Eidos.**

> _"From text to topology."_

## 🕸️ Overview

This directory contains the operational configuration and workspace for **Microsoft GraphRAG**, adapted for the Eidosian local environment. It is designed to ingest textual data from `input/`, process it using local LLMs, and generate a traversable knowledge graph.

## ⚙️ Configuration (`settings.yaml`)

The system uses **local inference** endpoints via Ollama.

- **Chat Model**: `qwen2.5:1.5b`
- **Embedding Model**: `nomic-embed-text:latest`
- **Endpoint**: `http://127.0.0.1:11434/v1`
- **Call timeout**: controlled in `settings.yaml` (`call_args.timeout`)

### Requirements
1. **Ollama** running:
```bash
ollama serve
```
2. Required models present:
```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text:latest
ollama list
```

## 🚀 Usage

### Indexing
Run the indexing pipeline to build the graph:

```bash
python -m graphrag index --root graphrag_workspace
```

### Querying
**Local Search** (Specific details):
```bash
python -m graphrag query --root graphrag_workspace --method local "Who is Annastasia?"
```

**Global Search** (Thematic summaries):
```bash
python -m graphrag query --root graphrag_workspace --method global "What are the core themes of the emails?"
```

## 🔗 Integration
This workspace is managed by `knowledge_forge`. The `grag_index` and `grag_query` tools in the Nexus map to these commands.
The MCP router defaults to this path (`graphrag_workspace`) and can be overridden with:

```bash
export EIDOS_GRAPHRAG_ROOT=/path/to/custom/graphrag_workspace
```
