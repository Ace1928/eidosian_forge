# GIS Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Map of Eidos.**

## 🗺️ Overview

`gis_forge` (Global Information System) is the centralized configuration registry for the Eidosian system.
It supports:
- **Hierarchical Keys**: Dot notation (`server.host`).
- **Environment Overrides**: `EIDOS_SERVER_HOST` overrides `server.host`.
- **Persistence**: Save/Load to JSON.
- **Pub/Sub**: Listen for config changes.

## 🏗️ Architecture
- `gis_core.py`: Main `GisCore` class.

## 🚀 Usage

```python
from gis_forge.gis_core import GisCore

gis = GisCore(persistence_path="config.json")
gis.set("server.port", 8080)
print(gis.get("server.port"))
```