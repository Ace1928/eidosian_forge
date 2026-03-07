# GIS Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Map of Eidos.**

## 🗺️ Overview

`gis_forge` (Grid Information System) is the master Ontology Manager and centralized configuration registry for the Eidosian ecosystem. 
It has evolved from a simple key-value store to managing the **Unified Eidosian Ontology (UEO)**—a high-performance, multidimensional vector graph that losslessly compresses Memory, Knowledge, Word, and Code into a single deductive substrate.

### Capabilities:
- **Unified Ontology (`ontology.py`)**: Vector-backed multidimensional graph storage (ChromaDB).
- **Hierarchical Config**: Dot notation (`server.host`).
- **Environment Overrides**: `EIDOS_SERVER_HOST` overrides `server.host`.

## 🏗️ Architecture
- `ontology.py`: The `UnifiedOntology` vector manager.
- `gis_core.py`: Legacy configuration manager.

## 🚀 Usage

```python
from gis_forge.gis_core import GisCore

gis = GisCore(persistence_path="config.json")
gis.set("server.port", 8080)
print(gis.get("server.port"))
```