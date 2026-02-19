# Code Forge Roadmap

## Current Line: v1.0.x (Archive Digester Foundation)

### Delivered
- Multi-language ingestion and unit normalization.
- Fingerprint-based deduplication (exact/normalized/near).
- Hybrid semantic search.
- Explainable triage pipeline with machine-readable artifacts.
- Knowledge Forge and GraphRAG export integration.
- Relationship-edge extraction (`imports`, `calls`, `uses`) and dependency graph artifact export.
- Benchmark suite with regression gates and baseline support.
- Canonical migration planning with staged compatibility shim generation.

### In Progress
- High-confidence canonical extraction workflow (old -> new module migration maps).
- Performance profiling and benchmark dashboards.
- Coverage expansion toward 90%+ on core ingestion/index/search paths.

## v1.1 (Canonicalization and Safety)
- Canonical module template generation for extracted code.
- Automated compatibility shims for moved APIs.
- Structural clone detection (AST shape + token features).
- Deletion gate policy:
  - no unique capability loss
  - tests green
  - benchmark non-regression

## v1.2 (Graph-Native Code Intelligence)
- Extended relationship graph (`imports`, `calls`, `uses`, `owns_test`).
- File/module coupling risk score.
- Redundancy pressure heatmaps for archive reduction.
- Cross-linking code claims into Knowledge Forge evidence chains.

## v1.3 (Living Code Substrate)
- Continuous digestion daemon mode with safe checkpoints.
- Drift detection for rules, schemas, naming conventions.
- Federated qualitative scoring adapters for model-assisted code review outputs.
- Automated change advisories into Agent Forge planning workflows.
