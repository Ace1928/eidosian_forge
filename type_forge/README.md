# Type Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Laws of Eidos.**

## 📏 Overview

`type_forge` provides runtime data validation.
It implements a lightweight schema engine (similar to JSON Schema) for validating dynamic data structures passed between agents.

## 🏗️ Architecture
- `src/type_forge/core.py`: Main validation engine.

## 🚀 Usage

```python
from type_forge import TypeCore

tc = TypeCore()
schema = {"type": "string", "minLength": 5}
tc.check_schema(schema, "Hello World")  # True
```

## GIS-backed Schema Registry

`TypeCore` can persist registered schemas via a GIS-compatible backend:

```python
from type_forge import TypeCore

tc = TypeCore(registry_backend=gis_core, registry_key="type_forge.schemas")
tc.register_schema("person", {"type": "object", "properties": {"name": {"type": "string"}}})
```
