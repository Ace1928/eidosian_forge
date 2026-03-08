# 🏷️ Metadata Forge ⚡

> _"The Labels of Eidos. Defining structure, semantics, and taxonomy."_

## 🧠 Overview

`metadata_forge` manages the unified schemas, taxonomies, and entity definitions used across the Eidosian ecosystem. It provides strict validation logic for tags and data shapes, ensuring that contextual metadata remains clean and cross-compatible between forges.

```ascii
      ╭───────────────────────────────────────────╮
      │             METADATA FORGE                │
      │    < Schemas | Validation | Taxonomy >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   JSON SCHEMAS      │   │   TAG MANAGER   │
      │ (Structural Rules)  │   │ (Semantic Cats) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Data Standardization & Validation
- **Test Coverage**: Core validation logic passing.
- **Architecture**:
  - `src/metadata_forge/schema/`: Schema definitions and JSON validation.
  - `src/metadata_forge/cli.py`: Minimal interface for checking schemas.

## 🚀 Usage & Workflows

### Python API

```python
from metadata_forge.schema.core import validate_tag

# Strict semantic validation
is_valid = validate_tag("ai_model")
print(f"Tag validity: {is_valid}")
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [ ] Implement central Pydantic schema registry for all `_forge` communications.

### Future Vector (Phase 3+)
- Migrate taxonomy logic directly into the Unified Eidosian Ontology (UEO) handled by `knowledge_forge`. `metadata_forge` will then pivot to become strictly a structural validation layer for MCP payloads.

---
*Generated and maintained by Eidos.*
