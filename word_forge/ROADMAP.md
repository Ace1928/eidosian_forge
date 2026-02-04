# Roadmap: word_forge

## Current Version: 0.6.0 (Beta)

---

## Phase 1: Foundation (v0.1.0 → v0.6.0) ✅ Complete

### Milestone 1.1: Core Infrastructure ✅
- [x] SQLite database layer
- [x] NetworkX graph foundation
- [x] NLTK/WordNet integration
- [x] Basic CLI interface

### Milestone 1.2: Enrichment Systems ✅
- [x] Emotion analysis (VADER, TextBlob)
- [x] Vector embeddings (sentence-transformers)
- [x] ChromaDB/FAISS backends
- [x] Graph visualization (PyVis, Plotly)

### Milestone 1.3: Processing Pipeline ✅
- [x] Background workers
- [x] Queue management
- [x] Batch processing
- [x] Comprehensive test suite

---

## Phase 2: Production Ready (v0.6.0 → v1.0.0) - Current

### Milestone 2.1: Code Quality
- [ ] Refactor config.py (modularize 44k LOC file)
- [ ] Improve type coverage (mypy strict)
- [ ] Performance profiling and optimization
- [ ] Memory usage optimization

### Milestone 2.2: Documentation
- [x] Create CURRENT_STATE.md
- [x] Create GOALS.md
- [x] Create ROADMAP.md
- [ ] Create ISSUES.md
- [ ] Create PLAN.md
- [ ] API documentation (Sphinx/MkDocs)

### Milestone 2.3: Integration
- [ ] MCP tool exposure via eidos_mcp
- [ ] Knowledge forge bridge
- [ ] Memory forge semantic layer
- [ ] Centralised config integration

---

## Phase 3: Enhanced Features (v1.0.0 → v2.0.0)

### Milestone 3.1: Advanced Graph
- [ ] Community detection
- [ ] Semantic path analysis
- [ ] Graph diffing and evolution
- [ ] Distributed graph storage

### Milestone 3.2: Multi-Language
- [ ] Language detection
- [ ] Cross-lingual embeddings
- [ ] Translation relationships
- [ ] Language-specific analyzers

### Milestone 3.3: Context Awareness
- [ ] Domain vocabularies
- [ ] Register detection
- [ ] Context-dependent meanings
- [ ] Disambiguation system

---

## Phase 4: AI Integration (v2.0.0 → v3.0.0)

### Milestone 4.1: LLM Enhancement
- [ ] LLM-powered relationship extraction
- [ ] Definition generation
- [ ] Example sentence synthesis
- [ ] Semantic validation

### Milestone 4.2: Active Learning
- [ ] Usage pattern learning
- [ ] Relationship discovery
- [ ] Quality self-assessment
- [ ] Continuous improvement

### Milestone 4.3: Reasoning
- [ ] Analogy completion
- [ ] Conceptual composition
- [ ] Inference over relationships
- [ ] Creative generation

---

## Timeline

| Phase | Target Date | Status |
|-------|-------------|--------|
| 1.1 | 2025-Q4 | ✅ Complete |
| 1.2 | 2025-Q4 | ✅ Complete |
| 1.3 | 2026-01 | ✅ Complete |
| 2.1 | 2026-02 | 🔶 In Progress |
| 2.2 | 2026-02 | 🔶 In Progress |
| 2.3 | 2026-03 | ⬜ Planned |
| 3.x | 2026-Q2 | ⬜ Future |
| 4.x | 2026-Q3 | ⬜ Future |

---

## Dependencies

### External
- `nltk` - Linguistic resources
- `networkx` - Graph operations
- `sentence-transformers` - Embeddings
- `chromadb` - Vector store
- `pyvis` - Graph visualization
- `plotly` - Interactive charts

### Internal
- `eidos_mcp` - Tool exposure
- `knowledge_forge` - Knowledge graph bridge
- `memory_forge` - Semantic memory

---

*Words are atoms. Graphs are molecules. Understanding is chemistry.*
