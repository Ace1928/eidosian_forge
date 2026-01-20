# Metadata Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Labels of Eidos.**

## 🏷️ Overview

`metadata_forge` manages the schemas and taxonomies used across the Eidosian system.
It defines:
- **Tags**: Standardized categories.
- **Schemas**: JSON Schemas for data validation.
- **Ontologies**: Relationships between concepts (lite version of `knowledge_forge`).

## 🏗️ Architecture
- `src/metadata_forge/`: Core logic.

## 🚀 Usage

```python
from metadata_forge import TagManager

tm = TagManager()
tm.validate_tag("ai") # True
```