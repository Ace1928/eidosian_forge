# Part 47: Phase 24 Deterministic Atlas Drift Guard

## Objective

Close the remaining documentation-operations gap from Phase 23 by making directory atlas generation reproducible, CI-enforced, and Termux/Linux portable:

1. Deterministic artifact output (no timestamp drift by default).
2. Explicit scope controls for tracked-only versus local filesystem views.
3. CI drift gate that fails stale atlas/index docs.
4. Full validation cycle with benchmark evidence.

## Implemented

### 1) Deterministic atlas/index generation controls

Files:

- `scripts/generate_directory_atlas.py`

Changes:

- Added `--scope tracked|filesystem` (default `tracked`).
- Added `--include-runtime-top-level` and `--include-hidden-top-level`.
- Added `--generated-at`:
  - default empty -> deterministic timestamp-free output,
  - `now` -> current UTC timestamp,
  - custom string -> explicit timestamp text.
- Split directory collection into tracked and filesystem collectors for predictable behavior.

### 2) Regression coverage expansion

Files:

- `scripts/tests/test_generate_directory_atlas.py`

Coverage additions:

- Tracked-scope atlas rendering and required section assertions.
- Hidden top-level inclusion/exclusion behavior for filesystem and runtime scans.
- `generated_at` resolution behavior (`""`, `"now"`, custom text).
- Deterministic index output shape with scope metadata.

### 3) CI drift gate workflow

Files:

- `.github/workflows/directory-atlas-drift.yml`
- `.github/workflows/README.md`

Behavior:

- Runs atlas unit tests.
- Regenerates `docs/DIRECTORY_ATLAS.md` and `docs/DIRECTORY_INDEX_FULL.txt` in tracked scope.
- Fails if regenerated artifacts differ from committed versions.

### 4) Docs entrypoint alignment

Files:

- `README.md`
- `docs/README.md`
- `scripts/README.md`

Updates:

- Regeneration commands now include explicit `--scope tracked`.
- Added optional local runtime command for filesystem scope.
- Script index updated to reflect deterministic/tracked default.

## Validation Executed

### Deterministic/idempotent atlas checks

- Regenerated atlas/index repeatedly in tracked scope and verified stable checksums across runs.

### Test suites

- `./eidosian_venv/bin/python -m pytest -q scripts/tests/test_generate_directory_atlas.py`
- `PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q scripts/tests/test_consciousness_benchmark_trend.py scripts/tests/test_linux_audit_matrix.py`
- `PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q agent_forge/tests/test_consciousness_*.py agent_forge/tests/test_events_corr.py`
- `PYTHONPATH=lib:agent_forge/src:eidos_mcp/src:crawl_forge/src ./eidosian_venv/bin/python -m pytest -q eidos_mcp/tests/test_mcp_tool_calls_individual.py`

### Runtime benchmark evidence

- Core benchmark:
  - `benchmark_20260217_040857_70f9c71c`
  - `reports/consciousness_benchmarks/benchmark_20260217_040857_70f9c71c.json`
- Stress benchmark:
  - `stress_20260217_040953_ec12fb11`
  - `reports/consciousness_stress_benchmarks/stress_20260217_040953_ec12fb11.json`
- Latest integrated benchmark retrieval:
  - `integrated_20260216_202514_6ce3fa3c`
  - `reports/consciousness_integrated_benchmarks/integrated_20260216_202514_6ce3fa3c.json`

## Notes

- Tracked scope is now the default for reproducible CI behavior.
- Filesystem scope remains available for local operational audits where non-tracked runtime paths matter.
- This phase is additive and backward compatible for previous atlas consumers.

## External References

- Python `argparse`:
  - https://docs.python.org/3/library/argparse.html
- Git diff plumbing:
  - https://git-scm.com/docs/git-diff
- Git tracked-file listing:
  - https://git-scm.com/docs/git-ls-files
