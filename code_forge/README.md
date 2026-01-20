# Code Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Architect.**

## 🛠️ Overview

`code_forge` analyzes, decomposes, and stores source code.
It is the engine behind the "Universal Repository" of Eidosian code snippets.

## 🏗️ Architecture
- **Analyzer**: Extracts structure (Classes, Functions) from source code using AST.
- **Librarian**: Manages a persistent store of code snippets (JSON/Vector).
- **Enricher**: (Planned) Uses LLMs to generate docs and tests for snippets.

## 🚀 Usage

```bash
# Analyze a file
code-forge analyze src/main.py

# Ingest into library
code-forge ingest src/main.py
```
