# File Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Hands of Eidos.**

## 📂 Overview

`file_forge` handles low-level file system interactions.
- **Reading/Writing**: Safe file I/O.
- **Searching**: Regex and glob pattern matching (`file_search`).
- **Management**: Creation, deletion, duplication detection.

## 🏗️ Architecture
- `file_core.py`: Main class `FileForge`.

## 🚀 Usage

```python
from file_forge.file_core import FileForge

ff = FileForge(base_path="/home/user")
matches = ff.search_content("TODO", directory="src")
```