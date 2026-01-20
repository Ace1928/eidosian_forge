# Code Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Eidosian Foundry for Source Code Manipulation.**

## 🛠️ Overview

`code_forge` is the domain responsible for understanding, modifying, and generating code.
It combines traditional static analysis (AST/CST) with modern LLM-driven synthesis.

## 🏗️ Architecture
- **Static Analysis**: Uses `libcst` and `rope` for safe, structural refactoring.
- **Polyglot Parsing**: Uses `tree-sitter` for understanding non-Python languages.
- **Legacy Engine**: Contains `forgeengine`, a narrative engine prototype (scheduled for migration).

## 📦 Capabilities
- **Refactoring**: Rename, Extract Method, Move Class.
- **Auditing**: Scan for patterns and anti-patterns.
- **Generation**: Scaffold boilerplate and tests.

## 🚀 Usage

```bash
# Analyze a file
codeforge analyze path/to/file.py

# Refactor a symbol
codeforge rename --from "OldName" --to "NewName" .
```
*(CLI implementation pending)*