# 📐 Type Forge ⚡

> _"The Laws of Eidos. Invariants enforced at the speed of thought."_

## 🧠 Overview

`type_forge` provides high-performance, dynamic runtime data validation for the Eidosian ecosystem. It implements a lightweight, native schema engine (conceptually similar to JSON Schema) designed specifically to validate dynamic data structures passed between autonomous agents without the overhead of heavy external frameworks.

```ascii
      ╭───────────────────────────────────────────╮
      │               TYPE FORGE                  │
      │    < Schema | Validation | Registry >     │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │ DYNAMIC VALIDATION  │   │ GIS PERSISTENCE │
      │ (Runtime Safety)    │   │ (Global Types)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Data Validation & Schema Enforcement
- **Test Coverage**: 100% Core Validation logic tested.
- **Architecture**:
  - `core.py`: Main `TypeCore` engine supporting validation and schema registration.

## 🚀 Usage & Workflows

### Python API

```python
from type_forge import TypeCore

tc = TypeCore()
schema = {"type": "string", "minLength": 5}
is_valid = tc.check_schema(schema, "Eidos")  # True
print(f"Validation result: {is_valid}")
```

### GIS-backed Schema Registry

`TypeCore` can persist registered schemas via the Global Information Store (GIS) to ensure agents share a coherent understanding of custom data shapes:

```python
from type_forge import TypeCore
from eidosian_core.gis import GISCore # Example global store

# Initialize with persistence
tc = TypeCore(registry_backend=GISCore(), registry_key="type_forge.schemas")

# Register a globally available schema
tc.register_schema("person", {
    "type": "object", 
    "properties": {
        "name": {"type": "string"}
    }
})
```

## 🔗 System Integration

- **Agent Forge**: Used by agents to rapidly assert the structural safety of payloads before acting on them.
- **MCP Nexus**: Validates input shapes to Eidos MCP Tools.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Implement robust error formatting (showing exact path of validation failure).

### Future Vector (Phase 3+)
- Integrate heavily with `metadata_forge` to form a unified Pydantic + Runtime Schema translation layer.

---
*Generated and maintained by Eidos.*
