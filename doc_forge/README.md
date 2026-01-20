# Doc Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Scribe of Eidos.**

## 📜 Overview

`doc_forge` automates the creation of API documentation and knowledge artifacts.
- **Extraction**: Reads Python docstrings.
- **Generation**: Creates Markdown or HTML output.
- **Validation**: Checks for missing documentation.

## 🏗️ Architecture
- `doc_core.py`: Main `DocForge` class wrapping `pdoc` or custom parsers.

## 🚀 Usage

```python
from doc_forge.doc_core import DocForge

df = DocForge(base_path="./src")
docs = df.generate_api_docs()
```