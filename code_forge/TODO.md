# Code Forge TODO

## Critical
- [x] Add ingestion/query benchmarks with reproducible fixtures and JSON outputs.
- [x] ~~Add coverage report target enforcement for `code_forge/src/code_forge`.~~
- [x] Implement canonical extraction generator + migration map artifact.
- [ ] Add signed manifests for roundtrip reconstruction/apply artifacts.
- [ ] Add OTLP trace exporter wiring from eval-run traces into external observability backends.

## High Priority
- [x] Add call/import/reference edge extraction to relationship graph.
- [x] Add triage confidence score and full rule audit trail output.
- [x] Add CLI command for benchmark execution and baseline comparison.
- [x] Add strict schema validation for digester artifacts.
- [x] Add reconstruct/parity/apply/roundtrip CLI flows with transactional backups.
- [x] Harden apply prune scope to managed reconstruction paths (prevent unmanaged deletion).
- [x] Add root-scoped integration export policy toggles (`run`, `effective_run`, `global`).
- [x] Add roundtrip contract validator (`validate-roundtrip`) with hash verification support.
- [x] Add provenance links artifact tying digester/roundtrip outputs to Knowledge/GraphRAG records.
- [x] Add Memory Forge sync and provenance linkage (`unit_id -> memory_id`) for digester/roundtrip.
- [x] Add eval operating system contracts + run matrix + replayable traces (`eval-init`, `eval-run`, `eval-staleness`).

## Medium Priority
- [x] Add AST-structural clone detection and cluster reports.
- [ ] Add per-language parser adapters (tree-sitter path) for deeper non-Python fidelity.
- [ ] Integrate profile traces into triage decisions (hot path preservation).
- [ ] Add parallelized parity hashing for large tree roundtrips.

## Low Priority
- [ ] Add web dashboard for triage and duplicate clusters.
- [ ] Add optional vector index backend for semantic retrieval experiments.
- [ ] Add archive reduction assistant output (candidate deletion PR plan).
