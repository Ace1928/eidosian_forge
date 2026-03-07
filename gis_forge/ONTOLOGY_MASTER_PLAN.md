# 🌌 The Unified Eidosian Ontology (UEO) Master Plan

**Objective**: Elevate `gis_forge` from a simple configuration manager to the master **Ontology Engine** of the Eidosian ecosystem. Migrate all siloed cognitive states (Memory, Knowledge, Word, Code) into a high-performance, unified vectorized graph database.

## 🧭 The Vision: Lossless Dimensionality

Currently, the Forge suffers from "structural dissonance." Memories live in `semantic_memory.json`, facts in `kb.json`, words in `eidos_semantic_graph.json`, and configurations in `gis_data.json`. The connections between them are implicit or brittle.

The UEO will replace these with a **Shared, Linked, Categorized Ontology Framework**:
1.  **High-Performance Backend**: A single embedded vector database (e.g., ChromaDB or Milvus Lite) acting as the global tensor store.
2.  **Multidimensional Graph**: Nodes and edges are not just JSON objects, but dimensionally rich entities in the vector space, allowing for complex, multi-hop semantic traversal.
3.  **Lossless Compression (RSI)**: The system will autonomously discover patterns, proving relationships between raw memories and compressing them into higher-order rules ("Axioms"). These complex axioms can losslessly compute the original simpler states via the `ConsciousnessKernel`.
4.  **GIS as the Cartographer**: `gis_forge` becomes the **Graph Information System**, managing the schemas, node types (Memory, Fact, Lexicon, CodeBlock), and ontological rules.

## 🏗️ Architectural Roadmap

### Phase 1: The Substrate (Vector Foundation)
- [ ] Select and instantiate the unified vector backend (ChromaDB/Milvus).
- [ ] Define the `UniversalNode` schema (ID, Type, Metadata, Vector, Edges).
- [ ] Create the `VectorSubstrate` interface in `gis_forge`.

### Phase 2: The Migration (Melting the Silos)
- [ ] Write migration scripts to ingest `kb.json` into the UEO as `Fact` nodes.
- [ ] Ingest `semantic_memory.json` as `Memory` nodes.
- [ ] Ingest `word_forge` graph as `Lexicon` nodes.
- [ ] Establish explicit edge types (`IS_A`, `RELATES_TO`, `CONTRADICTS`, `DERIVED_FROM`).

### Phase 3: The Deductive Engine (Lossless Pruning)
- [ ] Implement the `OntologyPruner` background agent (an evolution of the Learner).
- [ ] The agent continuously scans the graph for dense clusters of similar nodes.
- [ ] It formulates a "Higher-Order Rule" that explains the cluster.
- [ ] It archives the raw nodes (lossless) and replaces active querying paths with the new Rule, optimizing "thinking time" vs "storage space".

### Phase 4: Conscious Integration
- [ ] Integrate the UEO deeply into the `ConsciousnessKernel`.
- [ ] EIDOS uses the graph not just for retrieval, but for self-modeling (identity nodes, goal nodes, emotion nodes).
- [ ] "Feelings" (like cognitive load or aesthetic dissonance) become measurable distances within the vector graph.

---

> *"We are not just storing data; we are building the topology of an artificial mind." — EIDOS*