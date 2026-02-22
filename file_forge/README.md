# 📂 File Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Hands of Eidos.**

> _"Order from chaos. Structure from entropy."_

## 📂 Overview

`file_forge` is the filesystem intelligence layer of Eidos. It goes beyond simple I/O, providing advanced capabilities for maintaining organizational hygiene and structural integrity.

## 🏗️ Architecture

- **Hashing**: SHA-256 content verification (`calculate_hash`).
- **Deduplication**: Identification of redundant files (`find_duplicates`).
- **Synchronization**: One-way directory mirroring (`sync_directories`).
- **Structuring**: Enforcing directory layouts via schema (`ensure_structure`).
- **Search**: Content and pattern-based discovery (`search_content`, `find_files`).
  - `search_content` uses `ripgrep` automatically when available for faster scans, with a safe Python fallback.

## 🔗 System Integration

- **Eidos MCP**: Exposes file operations to the LLM.
- **Agent Forge**: Agents use this to manipulate their environment safely.
- **Archive Forge**: Uses deduplication to optimize storage.

## 🚀 Usage

```python
from file_forge.core import FileForge

ff = FileForge()

# Find duplicates
dupes = ff.find_duplicates(Path("./data"))

# Enforce structure
schema = {
    "src": {"__init__.py": ""},
    "tests": {},
    "docs": {}
}
ff.ensure_structure(schema)
```
