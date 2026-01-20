# Refactor Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Scalpel of Eidos.**

## 🔪 Overview

`refactor_forge` applies safe, structural transformations to code.
It uses `LibCST` to preserve comments and formatting while modifying the AST.

## 🏗️ Architecture
- `refactor_core.py`: Transformation engine.
- `analyzer.py`: Code complexity analysis.

## 🚀 Usage

```bash
refactor rename --from "OldClass" --to "NewClass" src/
```