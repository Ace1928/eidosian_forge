# Code Forge Plan

## Mission
Build a production-grade archive digester and living code substrate that can:
1. Index and normalize multi-language code.
2. Detect redundancy and semantic overlap deterministically.
3. Classify code into actionable triage buckets with explainable evidence.
4. Export proven artifacts into Knowledge Forge and GraphRAG.
5. Gate deletions/refactors on measurable tests and benchmarks.

## Stage Model (Archive Digester v1)

### Stage A: Intake (Deterministic Index)
- [x] Multi-language file discovery and hashing.
- [x] Deterministic `repo_index.json` artifact.
- [x] Per-file metadata: language, category, size, LOC, sha256, mtime.

### Stage B: Duplication and Similarity
- [x] Exact duplicate groups (`content_hash`).
- [x] Normalized duplicate groups (`normalized_hash`).
- [x] Near-duplicate pairs (`simhash64` + Hamming distance).
- [x] Hybrid semantic search (FTS + lexical fallback).

### Stage C: Explainable Triage
- [x] File-level metrics aggregation.
- [x] Rule-based labels: `keep`, `extract`, `refactor`, `quarantine`, `delete_candidate`.
- [x] Triage outputs: JSON + CSV + markdown report.

### Stage D: Integration Outputs
- [x] Knowledge Forge sync (`sync-knowledge`).
- [x] GraphRAG export (`export-graphrag`).
- [x] End-to-end digest command (`digest`) with optional integration exports.
- [x] Integration scope policy (`run`, `effective_run`, `global`) for deterministic export behavior.
- [x] Provenance links artifact (`provenance_links.json`) for cross-forge traceability.

### Stage E: Proof and Safety Gates
- [x] Expanded tests for similarity, multi-language analysis, triage pipeline.
- [x] Idempotent ingestion semantics.
- [x] Repeatable benchmark suite for ingestion/search/dependency graph plus regression gates.
- [x] Canonical migration map + compatibility shim staging artifacts.
- [x] Run-over-run drift report + immutable history snapshots.
- [ ] Coverage target >= 90% for `code_forge/src/code_forge`.
- [ ] Enforce deletion gate requiring tests + benchmark parity + migration map approvals.

### Stage F: Roundtrip Reconstruction and Auditability
- [x] Reconstruct source trees from `file_records` + `code_text` deterministically.
- [x] Generate parity report with file-level SHA256 checks.
- [x] Add transactional apply flow with backup and apply report artifacts.
- [x] Add one-shot `roundtrip` orchestration command for digest+integration+reconstruction+parity.
- [x] Add fallback integration scoping: when a run has zero new units, use latest effective run for source root.
- [x] Scope apply/prune operations to reconstruction-managed paths to prevent unmanaged file deletion.
- [x] Validate roundtrip on a second forge (`sms_forge`) with source/reconstructed test parity checks.
- [x] Add roundtrip contract validator (`validate-roundtrip`) with hash verification mode.
- [x] Add apply guardrails (`--require-manifest`, `--dry-run`) for safe promotions.
- [ ] Add deterministic regeneration ordering checks for large mixed-language trees (>10k files).
- [ ] Add signed artifact manifest option for tamper-evident promotion workflows.

## Near-Term Upgrades (v1.1)
- [x] Add symbol/reference graph edges beyond `contains` (`imports`, `calls`, `uses`).
- [x] Add structural clone detection (AST-shape similarity) for stronger near-dup filtering.
- [x] Add canonical extraction templates and compatibility shim generation.
- [x] Add triage confidence scoring and rule audit trace for every decision.
- [x] Add strict schema validation contract for digester artifacts.
- [x] Add reconstruction/parity/apply contracts for full project roundtrip validation.
- [ ] Add source-root scoped semantic retrieval index for faster large-forge regeneration queries.

## Done Definition
A Code Forge change is done only when:
1. New behavior is tested.
2. Artifacts are deterministic for the same commit + config.
3. CLI and docs are updated.
4. Integration paths (Knowledge Forge/GraphRAG) remain functional.
5. Regressions are measured with benchmarks, not guesses.
