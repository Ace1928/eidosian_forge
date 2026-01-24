# 🔮 Eidos MCP Installation

The Model Context Protocol server - the central nervous system of EIDOS.

## Prerequisites

- Python 3.10+
- Ollama (for local LLM)
- Apache Tika (optional, for document extraction)

## Quick Install

```bash
pip install -e ./eidos_mcp

# Start the MCP server
./run_server.sh
```

## Server Usage

```bash
# Run the MCP server
eidos-mcp serve --port 8000

# Check status
eidos-mcp status

# List available tools
eidos-mcp tools
```

## Configuration

Edit `config/models.py` for:
- LLM model selection
- Embedding model
- Timeout settings

## Dependencies

- `fastmcp` - MCP server framework
- `ollama_forge` - LLM client
- `memory_forge` - Memory system
- `eidosian_core` - Decorators and logging

