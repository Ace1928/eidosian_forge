# Code Forge / GIS Execution Program

Date: 2026-03-07
Status: Living implementation slice
Parent program: `docs/plans/EIDOSIAN_MASTER_IMPLEMENTATION_PROGRAM_2026-03-07.md`

## Intent

Turn Code Forge, archive reduction, GIS identity, and native GraphRAG into one coherent information substrate where:
- code units,
- extracted abstractions,
- archive artifacts,
- documents,
- references,
- lexicon terms,
- and memory/knowledge records

share stable identity, provenance, vector metadata, and graph relationships.

## Current Ground Truth

- Code Forge already has SQLite-backed storage, shared HNSW search, duplicate detection, provenance export, and graph export.
- Archive digester already emits provenance, triage, reduction-plan, and GraphRAG-oriented artifacts.
- Native GraphRAG is operational locally, but the lateral integration path between Code Forge outputs, GIS identity, and archive retirement gates is still incomplete.
- The next gain is integration and promotion policy, not a brand-new substrate.

## Primary Sources

Saved under `docs/external_references/2026-03-07-code-forge-gis/` and ingested into knowledge.

- SQLite WAL: `sqlite-wal.html`
- SQLite FTS5: `sqlite-fts5.html`
- SQLite WAL format: `sqlite-walformat.html`
- W3C PROV-O: `w3c-prov-o.html`
- W3C URI persistence: `w3c-uri-persistence.html`
- Tree-sitter intro: `tree-sitter-intro.html`
- Tree-sitter query syntax: `tree-sitter-query-syntax.html`
- HNSWlib README: `hnswlib-readme.md`
- Apache Tika: `apache-tika.html`

## Execution Tracks

### Track A: GIS Identity and Provenance
- [x] Define stable GIS identifiers for:
  - code units
  - source files
  - extracted abstractions
  - archive artifacts
  - document artifacts
  - lexicon terms
  - references
  - pipeline runs
- [x] Define deterministic serialization rules for GIS IDs.
- [x] Map Code Forge units and ingestion runs to GIS IDs.
- [x] Map archive triage, provenance links, and retirement gates to GIS IDs.
- [ ] Align GIS provenance fields with PROV-O concepts:
  - entity
  - activity
  - agent
  - wasDerivedFrom
  - wasGeneratedBy
  - used

### Track B: Code Forge Deep Upgrade
- [ ] Add canonical unit metadata contract:
  - GIS ID
  - provenance source
  - normalized abstraction fingerprint
  - archive origin
  - dependency summary
  - test linkage
  - reusable snippet summary
- [ ] Expand reverse lookup:
  - from abstraction to source files
  - from snippet to canonical unit
  - from file to extracted reusable units
  - from GraphRAG node to Code Forge unit
- [ ] Add stronger abstraction/reuse lanes:
  - exact duplicate collapse
  - structural clone clustering
  - semantic near-duplicate clustering
  - canonical abstraction candidate generation
- [ ] Add promotion gates before raw archive retirement.

### Track C: Archive Reduction
- [ ] Batch `archive_forge` into manageable ingestion waves.
- [ ] Separate:
  - code-like artifacts
  - document-like artifacts
  - metadata/provenance artifacts
  - binary/unsupported artifacts
- [ ] Route code-like artifacts through Code Forge first.
- [ ] Route document-like artifacts through doc/Tika/native GraphRAG.
- [ ] Retain deletion evidence and rollback pointers for every retirement action.

### Track D: Graph / Vector Integration
- [ ] Ensure every Code Forge unit stored in SQLite is mirrored into the shared vector substrate with stable metadata keys.
- [ ] Ensure every promoted code/document/archive artifact is mirrored into native GraphRAG with GIS identity.
- [ ] Add graph-side links for:
  - source file -> unit
  - unit -> abstraction
  - unit -> duplicate cluster
  - unit -> dependency
  - unit -> reference/doc
  - unit -> lexicon term
  - archive artifact -> promoted replacement
- [ ] Add rebuild/reconciliation jobs for stale vector metadata or missing graph edges.

### Track E: Validation and Retirement Gates
- [ ] Add per-batch archive ingestion artifact summaries.
- [ ] Add Code Forge reuse/retrieval benchmarks for archive-derived units.
- [ ] Add a retirement readiness report requiring:
  - promoted replacement exists
  - provenance preserved
  - reverse lookup preserved
  - tests/benchmarks pass
  - GraphRAG linkage exists
- [ ] Only then mark archive material removable.

## Immediate Implementation Queue

1. [x] Attach stable GIS/provenance identity fields to Code Forge units and ingestion artifacts.
2. [~] Add GraphRAG export reconciliation for Code Forge units, triage artifacts, and reduction-plan artifacts.
3. [~] Build batch classification and resumable ingestion state for `archive_forge`.
4. [ ] Add reverse lookup/report APIs needed by Atlas and the scheduler.
5. [ ] Add archive retirement gates and evidence artifacts.

## Progress Log

- [x] Primary-source set saved locally under `docs/external_references/2026-03-07-code-forge-gis/`.
- [x] Source set ingested locally via Tika-backed ingestion:
  - `files_processed=10`
  - `nodes_created=291`
- [x] Baseline Code Forge substrate reviewed:
  - `code_forge/src/code_forge/library/db.py`
  - `code_forge/src/code_forge/digester/pipeline.py`
- [x] Deterministic GIS identity added to the live Code Forge substrate:
  - `gis_forge/src/gis_forge/identity.py`
  - `code_forge/src/code_forge/library/db.py`
  - `code_forge/src/code_forge/integration/pipeline.py`
  - `code_forge/src/code_forge/integration/provenance.py`
  - `code_forge/src/code_forge/integration/provenance_registry.py`
- [x] Focused regression slice passed for GIS/Code Forge identity wiring:
  - `22 passed, 1 skipped`
- [x] Native GraphRAG artifact ingestion now reconciles Code Forge artifacts by `unit_gis_id`:
  - `knowledge_forge/src/knowledge_forge/integrations/graphrag.py`
  - `knowledge_forge/tests/test_kb.py`
- [x] Extended focused regression slice passed for GIS + GraphRAG reconciliation:
  - `32 passed, 2 skipped`
- [x] Archive batch planning and resumable ingestion state scaffold added:
  - `code_forge/src/code_forge/digester/pipeline.py`
  - `code_forge/src/code_forge/digester/__init__.py`
  - `code_forge/tests/test_digester_pipeline.py`
- [x] Extended focused regression slice passed for digester + GIS + GraphRAG:
  - `37 passed, 2 skipped`
