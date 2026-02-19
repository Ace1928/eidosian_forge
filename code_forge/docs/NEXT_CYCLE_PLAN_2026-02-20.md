# Code Forge Next Cycle Plan (Cycle 07)

Date: 2026-02-20  
Baseline: Roundtrip validated on `audit_forge` + `sms_forge` with managed-scope apply safety.

## Cycle Goals
1. Hard-fail unsafe apply conditions and introduce preflight contract validation.
2. Add configurable integration export policy modes (`run`, `effective_run`, `global`) with strict schema output.
3. Expand deterministic regeneration checks and benchmark/scalability telemetry.

## Phase 1: Roundtrip Contract Hardening
- [ ] Add `validate-roundtrip` CLI command.
- [ ] Validate:
  - `reconstruction_manifest.json`
  - `parity_report.json`
  - `roundtrip_summary.json`
  - backup `apply_report.json` (when apply used)
- [ ] Add schema file/module for roundtrip artifacts with machine-readable error output.
- [ ] Fail apply if reconstruction manifest is missing and `--require-manifest` is set.
- [ ] Add optional `--dry-run-apply` preview mode.

### Acceptance
- `validate-roundtrip` returns pass/fail with actionable error list.
- Invalid artifact structure fails CI test fixtures.

## Phase 2: Integration Policy Controls
- [ ] Add CLI flag: `--integration-policy {run,effective_run,global}`.
- [ ] Wire policy into digest + roundtrip command paths.
- [ ] Ensure summaries explicitly record selected policy and resolved `integration_run_id`.
- [ ] Add tests for each policy mode.

### Acceptance
- Policy behavior is deterministic and test-covered.
- `global` mode preserves backward-compatible behavior.

## Phase 3: Determinism + Scale
- [ ] Add deterministic ordering assertion for large manifest generation.
- [ ] Add optional parallel hashing (`--jobs`) for parity checks.
- [ ] Add benchmark fixture for medium forge roundtrip (timings + memory).
- [ ] Add regression gate thresholds for:
  - parity runtime,
  - reconstruction runtime,
  - apply runtime.

### Acceptance
- Same inputs produce byte-identical manifest ordering.
- Benchmarks output structured JSON and regression pass/fail status.

## Phase 4: Multi-Forge Promotion Path
- [ ] Validate roundtrip on `diagnostics_forge` and `article_forge` using hardened contracts.
- [ ] Produce per-forge migration/promotion report with parity + test parity evidence.
- [ ] Enforce replacement gate:
  - parity pass,
  - source tests pass,
  - reconstructed tests pass,
  - apply report clean (no unmanaged deletions).

### Acceptance
- Promotion reports are auditable and reproducible.
- No manual assumptions required for replacement decision.

## Risks and Controls
- Risk: unmanaged file deletion during apply.
  - Control: managed-scope apply + validate-roundtrip + dry-run.
- Risk: policy confusion between run/effective/global.
  - Control: explicit policy in CLI and summary payload.
- Risk: silent schema drift.
  - Control: contract validation tests + CI job.

## Deliverables
- New CLI subcommands/flags.
- Expanded test suite for contracts/policy/scale.
- Benchmark and promotion reports under `code_forge/docs/`.
- Updated `PLAN.md`, `TODO.md`, `CURRENT_STATE.md`, `README.md`, `CHANGELOG.md`.
