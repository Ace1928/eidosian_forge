# Eidos MCP Server

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Central Nervous System.**

## 🧠 Overview

`eidos_mcp` is the implementation of the [Model Context Protocol](https://github.com/model-context-protocol/mcp) server.
It exposes the capabilities of all other Forges (Memory, Knowledge, Coding, etc.) as **Tools** and **Resources** to the LLM.

## 🔗 Integrations
- **Memory**: `memory_add`, `memory_retrieve`
- **Knowledge**: `kb_add`, `grag_query`
- **System**: `run_shell_command`, `file_read`, `file_write`
- **Audit**: `audit_mark_reviewed`, `audit_add_todo`

## 🚀 Usage

```bash
# Start the server (std/stdio mode)
python eidos_mcp_server.py
```

## 🛠️ Configuration
Configuration is loaded from the **GIS** (Global Info System) or `config.json`.