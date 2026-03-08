# 🗺️ GIS Forge ⚡

> _"The Map of Eidos. Grid Information System and Centralized Configuration."_

## 🧠 Overview

`gis_forge` is the definitive repository for the Eidosian Framework's settings, hierarchical configurations, and structural metadata. It manages the **Unified Eidosian Ontology (UEO)** substrate, serving as the high-performance source of truth for both static system parameters and dynamic vector-backed semantic relationships.

```ascii
      ╭───────────────────────────────────────────╮
      │                GIS FORGE                  │
      │    < Configuration | UEO | Persistence >  │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   GLOBAL STORE      │   │  VECTOR SUBSTRATE│
      │ (Key-Value Config)  │   │ (HNSWlib / UEO)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Configuration & Ontology Management
- **Test Coverage**: Core config and vector substrate logic verified.
- **MCP Integration**: 4 Tools (`gis_get`, `gis_restore`, `gis_set`, `gis_snapshot`).
- **Core Components**:
  - `core.py`: Main configuration manager with dot-notation support.
  - `vector_substrate.py`: High-performance HNSWlib-backed storage for UEO vectors.
  - `ontology_engine.py`: Logic for mapping semantic nodes into the GIS grid.

## 🚀 Usage & Workflows

### Configuration Management (Python)

```python
from gis_forge.core import GISCore

gis = GISCore(persistence_path="gis_data.json")

# Hierarchical setting
gis.set("agent.timeout", 300)

# Environment-aware retrieval
port = gis.get("server.port", default=8080)
```

### UEO Interaction (CLI)

```bash
# Snapshot the current GIS state
python -m gis_forge.cli snapshot --tag "stable_v1"

# Retrieve a specific configuration value
python -m gis_forge.cli get "agent.default_model"
```

## 🔗 System Integration

- **Eidos MCP**: Central registry for all tool-enabling environment variables.
- **Knowledge Forge**: Leverages the GIS vector substrate to store persistent embeddings of semantic concepts.
- **Diagnostics Forge**: Stores threshold definitions for anomaly detection.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Stabilize HNSWlib-backed UEO substrate.

### Future Vector (Phase 3+)
- Implement "Cross-Agent Config Sync" where multiple instances of the Eidosian runtime can negotiate shared GIS parameters via a decentralized gossip protocol.

---
*Generated and maintained by Eidos.*
