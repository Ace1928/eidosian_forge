# Runtime Artifact Policy

## Purpose

This policy separates source-of-truth code and documentation from generated runtime state.
The forge now has enough autonomous and benchmark activity that treating runtime artifacts as normal source files creates review noise, rebase churn, and false regressions.

## What Must Stay Out Of Source Control

These paths are generated and should be ignored by default:

- transient files under `data/**/*.tmp`
- vector indexes under `data/**/vectors/index.bin`
- runtime logs and journals under `data/runtime/**/*.jsonl`
- generated status/history files under `data/runtime/*_status.json` and `data/runtime/*_history.json`
- external benchmark working trees under `data/runtime/external_benchmarks/**`
- fetched external source workspaces under `data/runtime/external_sources/**`

## What Should Be Preserved

Curated artifacts belong under stable, reviewable locations such as:

- `reports/` for benchmark and proof outputs intended for comparison or publication
- `docs/` for references, plans, and operational doctrine
- source packages under `*/src/` and validating tests under `*/tests/`

## Audit Mechanism

Use:

```bash
PYTHONPATH=lib ./eidosian_venv/bin/python scripts/runtime_artifact_audit.py \
  --output reports/runtime_artifact_audit/latest.json
```

The audit reports:

- tracked generated files that should be removed from version control over time
- live generated files currently present in the workspace
- recommended cleanup direction

## Current Standard

The current standard is intentionally incremental:

1. stop new generated artifacts from entering source control
2. identify legacy tracked runtime artifacts through the audit
3. migrate durable evidence into `reports/` and durable doctrine into `docs/`
4. only then untrack legacy generated runtime files in controlled batches

## Rationale

This keeps the forge reproducible without confusing runtime products with source.
It also reduces rebase pressure, makes CI diffs legible, and keeps long-running autonomous services from polluting ordinary code review.
