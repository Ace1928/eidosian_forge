# Pipeline Overview

The repository provides a single command to run the full verification suite and emit artifacts.

## Verification entrypoint

```bash
scripts/verify.sh
```

## Artifacts
The verification script writes:
- `artifacts/unit-tests.xml`: unit JUnit report
- `artifacts/integration-tests.xml`: integration JUnit report
- `artifacts/profile.pstats`: cProfile stats
- `artifacts/benchmark.json`: benchmark suite (multiple named benchmarks)
- `artifacts/index.json`: indexed metadata including test, profile, and benchmark summaries
- `artifacts/index.db`: SQLite ingestion database
- `artifacts/report.json`: performance and test trend report

The verification script prefers `.venv/bin/python` when available.

## Optional tooling
`ruff` and `mypy` run automatically when installed; otherwise they are skipped with a warning.
