# 🧠 Memory Forge ⚡

> _"The Stream of Eidos. Tiered, semantic, and persistent episodic recall."_

## 🧠 Overview

`memory_forge` provides the episodic memory layer for Eidosian agents. It implements a multi-tiered storage architecture (Working, Short-Term, Long-Term) with semantic compression and vector-backed retrieval. It ensures that every interaction, thought, and sensory input can be retrieved based on relevance and importance.

```ascii
      ╭───────────────────────────────────────────╮
      │               MEMORY FORGE                │
      │    < Episodes | Tiers | Compression >     │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   VECTOR RECALL     │   │ SEMANTIC COMP.  │
      │ (HNSW / Semantic)   │   │ (Context Guard) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Episodic Memory & Vector Search
- **Test Coverage**: Core tiered-memory and JSON storage verified.
- **MCP Integration**: 11 Tools (`memory_add`, `memory_retrieve`, `tiered_remember`, etc.).
- **Architecture**:
  - `core/tiered_memory.py`: Manages importance-weighted promotion between memory tiers.
  - `backends/json_store.py`: Robust, local-first episodic persistence.
  - `compression/`: Logic for merging redundant memory traces to save context window space.

## 🚀 Usage & Workflows

### Episodic Memory (Python)

```python
from memory_forge.core.main import MemoryForge

memory = MemoryForge()

# Record a new interaction
memory.add("The user preferred 4-space indentation for all Python scripts.")

# Semantic retrieval
hits = memory.retrieve("indentation preferences")
for hit in hits:
    print(f"[{hit.score:.2f}] {hit.content}")
```

### Tiered Persistence (MCP)

```bash
# Add an important fact to 'working' memory
tiered_remember --content "UEO Migration phase 1 is locked." --importance 0.9 --tier working

# Recall from all tiers
tiered_recall --query "migration status" --tier all
```

## 🔗 System Integration

- **Agent Forge**: Provides the immediate "History" and "Context" for reasoning loops.
- **Knowledge Forge**: Consumes promoted memories to form permanent facts in the semantic graph.
- **Diagnostics Forge**: Logs system-critical state events into episodic memory for historical triage.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Migrate defaults to canonical 768-dimension vectors.
- [ ] Implement "Context Rescue" daemon to auto-archive low-importance memories during context window pressure.

### Future Vector (Phase 3+)
- Transition episodic storage to a fully encrypted block-chain style ledger to guarantee the immutability of agent history.

---
*Generated and maintained by Eidos.*
