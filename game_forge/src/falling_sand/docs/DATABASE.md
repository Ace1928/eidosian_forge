# Database Ingestion

The pipeline can ingest `artifacts/index.json` into a SQLite database for long-term storage and querying.

## Schema
Tables are versioned through `schema_migrations` and include:
- `runs`
- `entries`
- `test_summary` + `test_cases`
- `profile_summary` + `profile_functions`
- `benchmark_summary`

## Usage

```bash
python scripts/ingest_index.py --index artifacts/index.json --db artifacts/index.db
falling-sand-ingest --index artifacts/index.json --db artifacts/index.db
```

To control batching, pass `--batch-size` (default: 1000).

## Reporting
Generate a trend report from the database:

```bash
python -m falling_sand.reporting --db artifacts/index.db --output artifacts/report.json
```

## Migrations
Database schema migrations are applied automatically on ingestion. Add new migrations by extending
`falling_sand.db._migrations()` and bumping `CURRENT_DB_VERSION`.

## Performance tuning
SQLite connection pragmas are configurable through `DbConfig`:
- `journal_mode` defaults to `WAL`
- `synchronous` defaults to `NORMAL`
- `cache_size_kb` defaults to `20000` (negative value is used to specify KB)
- `temp_store` defaults to `MEMORY`
- `busy_timeout_ms` defaults to `5000`
- `batch_size` defaults to `1000` for bulk inserts

These defaults favor performance while preserving safety for the ingestion workload.

Benchmarks are stored in `benchmark_cases` with per-benchmark rows keyed by name.
