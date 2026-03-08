# 📂 File Forge ⚡

> _"The Hands of Eidos. Order from chaos. Structure from entropy."_

## 🧠 Overview

`file_forge` is the filesystem intelligence layer of Eidosian Forge. It goes beyond simple I/O, providing advanced capabilities for maintaining organizational hygiene, structural integrity, and deep content discovery.

```ascii
      ╭───────────────────────────────────────────╮
      │               FILE FORGE                  │
      │    < Sync | Deduplicate | Structure >     │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   HASH FINGERPRINTS │   │   RIPGREP SCAN  │
      │ (SHA-256)           │   │ (Content Match) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Filesystem Intelligence
- **Test Coverage**: 100% Core Logic Validated
- **Architecture**:
  - **Hashing**: SHA-256 content verification (`calculate_hash`).
  - **Deduplication**: Identification of redundant files via content hash (`find_duplicates`).
  - **Synchronization**: Intelligent one-way directory mirroring based on hash equality (`sync_directories`).
  - **Structuring**: Enforcing complex directory layouts via declarative schema (`ensure_structure`).
  - **Search**: Content (`search_content`) and glob pattern (`find_files`) discovery. Auto-detects and uses `ripgrep` (`rg`) for high-velocity scans when available, with a safe pure-Python fallback.

## 🚀 Usage & Workflows

### Python API

```python
from pathlib import Path
from file_forge.core import FileForge

ff = FileForge()

# Find exact duplicates in a directory
dupes = ff.find_duplicates(Path("./data"))
for file_hash, paths in dupes.items():
    print(f"Hash {file_hash[:8]} has {len(paths)} copies.")

# Enforce declarative structure
schema = {
    "src": {"__init__.py": ""},
    "tests": {},
    "docs": {}
}
ff.ensure_structure(schema, root=Path("./my_new_project"))

# High-velocity content search
results = ff.search_content("TODO: fix this", directory=Path("./src"))
```

## 🔗 System Integration

- **Eidos MCP**: Exposes intelligent file operations to the cognitive layer.
- **Agent Forge**: Provides safe manipulation primitives for embedded agents.
- **Archive Forge**: Supplies deduplication logic for storage reduction.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Integrate `ripgrep` backend for extreme search velocity.

### Future Vector
- Integrate direct links to `knowledge_forge` when structuring known repositories.
- Extend `sync_directories` to support delta patches rather than full file overwrites.

---
*Generated and maintained by Eidos.*
