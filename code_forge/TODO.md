# Code Forge TODO

## Critical
- [ ] Add ingestion/query benchmarks with reproducible fixtures and JSON outputs.
- [ ] Add coverage report target enforcement for `code_forge/src/code_forge`.
- [ ] Implement canonical extraction generator + migration map artifact.

## High Priority
- [ ] Add call/import/reference edge extraction to relationship graph.
- [ ] Add triage confidence score and full rule audit trail output.
- [ ] Add CLI command for benchmark execution and baseline comparison.
- [ ] Add strict schema validation for digester artifacts.

## Medium Priority
- [ ] Add AST-structural clone detection and cluster reports.
- [ ] Add per-language parser adapters (tree-sitter path) for deeper non-Python fidelity.
- [ ] Integrate profile traces into triage decisions (hot path preservation).

## Low Priority
- [ ] Add web dashboard for triage and duplicate clusters.
- [ ] Add optional vector index backend for semantic retrieval experiments.
- [ ] Add archive reduction assistant output (candidate deletion PR plan).
