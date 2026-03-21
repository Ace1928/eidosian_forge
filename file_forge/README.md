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

## 🧬 Eidosian Substrate

`file_forge` now has a reversible file-library substrate alongside the utility API:
- SQLite-backed blob and metadata storage for arbitrary files
- deterministic vector rows for lightweight similarity/search surfaces
- forge links describing how a file should bridge into `code_forge`, `knowledge_forge`, `word_forge`, and `memory_forge`
- reversible restore from indexed blobs
- incremental indexing that skips unchanged files
- optional `ingest_and_remove` style operation via `--remove-after-ingest`
- subtree restoration via `file-forge restore-tree`
- archive-lifecycle participation as the reversible substrate for non-code and off-plan files

This is now the byte-faithful file substrate used by the archive lifecycle alongside Code Forge rather than only helper functions.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Integrate `ripgrep` backend for extreme search velocity.

### Future Vector
- Integrate direct links to `knowledge_forge` when structuring known repositories.
- Extend `sync_directories` to support delta patches rather than full file overwrites.

---
*Generated and maintained by Eidos.*

## Unified Role

- `word_forge`: living lexicon and term graph
- `memory_forge`: living memory and continuity substrate
- `knowledge_forge`: living facts, lessons, and evidence graph
- `code_forge`: living code abstraction and provenance substrate
- `file_forge`: living file and byte-faithful reversibility substrate
- `scribe` / `doc_forge`: living documentation processor
- `atlas`: operator-facing unified control plane

In the current contract, File Forge is the layer that makes full-source-tree retirement possible even when the Code Forge plan is intentionally narrower than the repository filesystem.
