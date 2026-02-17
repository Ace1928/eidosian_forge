# 🗄️ Eidosian Data

The persistent state and knowledge artifacts of the system.

## 🗂️ Directory Structure

### Knowledge & Memory
- **`kb.json`**: The primary Knowledge Base node store.
- **`tiered_memory/`**: Multi-tier memory snapshots (self, user, context).
- **`episodic_memory.json` / `semantic_memory.json`**: Legacy or flat-file memory stores.
- **`lexical_graph.gexf`**: Graph representation of the Eidosian lexicon.

### Forge-Specific Data
- **`code_forge/`**: AST indices and snippet libraries.
- **`moltbook/`**: Cached posts, social graphs, and interest scores.
- **`consciousness/`**: Causal traces and protocol state.

### Databases
- **`db/`**: SQLite databases for time-series metrics and agent journals.
- **`word_forge.sqlite*`**: Relational store for the semantic lexicon.

## ⚠️ Integrity Note
Data files under this directory are updated automatically by running Forges. Manual editing is discouraged unless performing critical state recovery.
