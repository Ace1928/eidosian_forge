# Reporting

The reporting pipeline aggregates historical performance and test data stored in SQLite.

## Output
`artifacts/report.json` includes:
- `runs`: recent run metadata
- `benchmark_trends`: per-benchmark mean series over time
- `test_trend`: test counts and duration over time
- `hotspots`: top aggregate profile hotspots across runs

## Usage

```bash
python -m falling_sand.reporting --db artifacts/index.db --output artifacts/report.json
```

## Options
- `--run-limit`: number of recent runs to include (default: 20)
- `--top-n`: number of hotspot functions to include (default: 10)
