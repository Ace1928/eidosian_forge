# 🧠 Knowledge Forge ⚡

> _"The Memory of Eidos. Persistent semantic structures and graph-based reasoning."_

## 🧠 Overview

`knowledge_forge` is the persistent knowledge graph layer for the Eidosian ecosystem. It manages semantic relationships between concepts, providing a deductive substrate that bridges episodic memory with permanent factual storage. It integrates deeply with **GraphRAG** for global/local querying and includes advanced reasoning capabilities via OWL/RDF support.

```ascii
      ╭───────────────────────────────────────────╮
      │             KNOWLEDGE FORGE               │
      │    < Graph | Search | GraphRAG | OWL >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   SEMANTIC GRAPH    │   │  MEMORY BRIDGE  │
      │ (Nodes & Edges)     │   │ (Working Memory)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Active
- **Type**: Semantic Knowledge Management
- **Test Coverage**: Core graph logic and GraphRAG fallbacks verified.
- **MCP Integration**: 
  - 7 KB Tools (`kb_add`, `kb_search`, `kb_link`, etc.)
  - 6 GraphRAG Tools (`grag_query`, `grag_index`, etc.)
- **Architecture**:
  - `core/graph.py`: Main `KnowledgeForge` engine for node/link management.
  - `core/bridge.py`: `KnowledgeMemoryBridge` for unified semantic search.
  - `integrations/graphrag.py`: Compatibility layer for external GraphRAG tools.

## 🚀 Usage & Workflows

### Semantic Search

```bash
# General status and statistics
knowledge-forge status

# Search across the knowledge graph
knowledge-forge search "quantum entanglement"

# Unified search (Knowledge + Episodic Memory)
knowledge-forge unified "recent tasks"
```

### GraphRAG Interaction (MCP)

```bash
# Trigger an incremental index run
# (Combines native vector indexing with external GraphRAG if available)
grag_index --scan_roots ["./src", "./docs"]

# Execute a global thematic query
grag_query --query "What are the core Eidosian principles?"
```

## 🔗 System Integration

- **Agent Forge**: Provides the "Long-Term Memory" substrate for autonomous reasoning.
- **Code Forge**: Maps code unit provenance into the knowledge graph for structural traceability.
- **Word Forge**: Feeds the underlying semantic lexicon required for concept disambiguation.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Stabilize HNSWlib dimension alignment (canonical 768).
- [ ] Implement automated knowledge "Cleanup" agents to prune weak or redundant edges.

### Future Vector (Phase 3+)
- Transition to a fully distributed knowledge graph supporting cross-instance synchronization.
- Implement native Neuro-Symbolic reasoning loops directly on the graph substrate.

---
*Generated and maintained by Eidos.*
