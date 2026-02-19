# Code Forge Roundtrip Validation: `sms_forge`

Date: 2026-02-19  
Cycle IDs: `sms_forge_cycle05` (initial), `sms_forge_cycle06` (safeguard-validated rerun)

## Objective
- Run full Code Forge ingest -> normalize -> integration sync -> reconstruct -> parity -> apply on a second forge.
- Prove compatibility parity by executing `sms_forge` tests in both source and reconstructed trees.
- Harden apply semantics to prevent deletion outside managed reconstruction scope.

## Baseline Reliability Fix
Before roundtrip, `sms_forge` tests had a Python 3.12 incompatibility (`asyncio.coroutine` removed).

Fix:
- Updated `sms_forge/tests/test_sms_core.py` to use `AsyncMock` for async subprocess mock.
- Baseline tests now pass (`6/6`).

## Roundtrip Execution (Initial)
```bash
PYTHONPATH=lib:code_forge/src ./eidosian_venv/bin/python -m code_forge.cli --json \
  roundtrip sms_forge \
  --workspace-dir data/code_forge/roundtrip/sms_forge_cycle05 \
  --mode analysis \
  --sync-knowledge \
  --export-graphrag \
  --graphrag-output-dir data/code_forge/roundtrip/sms_forge_cycle05/graphrag \
  --graph-export-limit 500 \
  --apply \
  --backup-root Backups/code_forge_roundtrip
```

Observed issue in `cycle05`:
- Parity passed, but apply pruned files outside managed scope (`.gitignore`, `CODEOWNERS`, workspace file).
- Root cause: apply compared full target tree without scope filtering.

## Safeguard Implementation
Applied fix in `code_forge/src/code_forge/reconstruct/pipeline.py`:
- `apply_reconstruction(...)` now scopes changes/removals using:
  - explicit `managed_relative_paths` when provided, or
  - `reconstruction_manifest.json` entries, or
  - extension scope fallback.
- `run_roundtrip_pipeline(...)` now passes managed paths from reconstruction manifest entries.
- Added regression test:
  - `test_apply_reconstruction_prune_only_managed_scope`

## Roundtrip Execution (Safeguard Rerun)
```bash
PYTHONPATH=lib:code_forge/src ./eidosian_venv/bin/python -m code_forge.cli --json \
  roundtrip sms_forge \
  --workspace-dir data/code_forge/roundtrip/sms_forge_cycle06 \
  --mode analysis \
  --sync-knowledge \
  --export-graphrag \
  --graphrag-output-dir data/code_forge/roundtrip/sms_forge_cycle06/graphrag \
  --graph-export-limit 500 \
  --apply \
  --backup-root Backups/code_forge_roundtrip
```

## Cycle06 Result Summary
- `parity_pass`: `true`
- source/reconstructed files in scope: `32 / 32`
- hash mismatches: `0`
- apply: `noop=true`
- apply removed files: `0`
- knowledge sync:
  - `scanned_units=183`
  - `created_nodes=0`
  - `existing_nodes=97`
- GraphRAG export:
  - `exported=97`
  - `skipped=86`
  - language split: `python=85`, `markdown=8`, `external=3`, `toml=1`

## Functional Parity Validation
Source tests:
```bash
PYTHONPATH=lib:sms_forge/src ./eidosian_venv/bin/python -m pytest sms_forge/tests -q
```
Result: `6 passed`

Reconstructed tests:
```bash
PYTHONPATH=lib:data/code_forge/roundtrip/sms_forge_cycle06/reconstructed/src \
  ./eidosian_venv/bin/python -m pytest \
  data/code_forge/roundtrip/sms_forge_cycle06/reconstructed/tests -q
```
Result: `6 passed`

## Key Artifacts
- Roundtrip summary:
  - `data/code_forge/roundtrip/sms_forge_cycle06/roundtrip_summary.json`
- Parity report:
  - `data/code_forge/roundtrip/sms_forge_cycle06/parity_report.json`
- Reconstruction manifest:
  - `data/code_forge/roundtrip/sms_forge_cycle06/reconstructed/reconstruction_manifest.json`
- Digester summary:
  - `data/code_forge/roundtrip/sms_forge_cycle06/digester/archive_digester_summary.json`

## Contract Outcome
- End-to-end ingestion/regeneration pipeline validated on second forge.
- Apply semantics are now managed-scope safe and idempotent.
- Functional parity verified by test execution in both source and regenerated trees.
