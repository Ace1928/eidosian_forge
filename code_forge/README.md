# 🏗️ Code Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Architect of Structure.**

> _"Code is thought frozen in silicon. We thaw it, reshape it, and freeze it anew."_

## 🛠️ Overview

`code_forge` is the static analysis and code manipulation engine of Eidos. It deconstructs source code into its constituent parts (AST, Control Flow, Dependencies) to enable intelligent refactoring and understanding.

## 🏗️ Architecture

- **Analyzer**: Extracts semantic structure (Classes, Functions, Imports) using Python's `ast` module.
- **Librarian**: Manages the "Universal Repository" of code snippets, indexing them for retrieval.
- **Refactorer**: Provides tools for safe, programmatic code modification (used by `refactor_forge`).

## 🔗 System Integration

- **Documentation Forge**: `doc_forge` uses `code_forge` utilities to parse file structures before documenting them.
- **Agent Forge**: Agents use `code_forge` tools to "read" code with structural awareness.

## 🚀 Usage

### CLI

```bash
# Analyze a file structure
code-forge analyze src/main.py

# Ingest into library
code-forge ingest src/main.py

# Build/refresh SQLite structural index from a directory
code-forge ingest-dir .

# Report duplicate code units by normalized content hash
code-forge dedup-report --min-occurrences 2 --limit-groups 50

# Trace hierarchical contains-graph for a unit
code-forge trace module.path.ClassName --depth 3 --max-nodes 200
```

### Python API

```python
from code_forge import Analyzer

analyzer = Analyzer()
structure = analyzer.parse_file("src/main.py")
print(structure.classes)
```
