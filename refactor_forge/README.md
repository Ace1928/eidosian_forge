# 🔪 Refactor Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Scalpel of Eidos.**

> _"Precision modification without loss of intent."_

## 🔪 Overview

`refactor_forge` provides structural code transformation capabilities. It operates on the Python Abstract Syntax Tree (AST) to perform safe, automated refactorings that maintain semantic correctness while optimizing structure.

## 🏗️ Architecture

- **Core Engine (`__init__.py`)**: Implements the `RefactorForge` class, wrapping Python's `ast` and `ast.NodeTransformer`.
- **Rename Transformer**: Idempotent identifier renaming across functions and classes.
- **Docstring Stripper**: Automatic removal of documentation artifacts for code compression or privacy tasks.
- **Analyzer (`analyzer.py`)**: Metrics-based code quality assessment.

## 🔗 System Integration

- **Eidos MCP**: Exposes `refactor_analyze` and `refactor_transform` tools.
- **Agent Forge**: Agents use these tools to programmatically improve their own source or resolve complex merge conflicts.

## 🚀 Usage

### Python API

```python
from refactor_forge import RefactorForge

rf = RefactorForge()
transformed_code = rf.transform(
    source="def old_func(): pass",
    rename_map={"old_func": "new_func"},
    remove_docs=True
)
print(transformed_code)
# Output: def new_func():\n    pass
```

### CLI

```bash
python -m refactor_forge.cli path/to/source.py --analyze-only
python -m refactor_forge.cli path/to/source.py --dry-run --preview-diff-against path/to/proposed.py
```
