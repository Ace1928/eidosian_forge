# Indexing

The indexer scans Python source and test files to emit structured metadata.
It uses `os.scandir` for fast directory traversal and avoids following symlinks by default.

## Output schema
The JSON payload includes:
- `schema_version`: schema version identifier
- `generated_at`: ISO-8601 timestamp
- `source_root` / `tests_root`: scanned roots
- `stats`: summary counts
- `entries`: symbol records containing name, qualname, kind, origin, module, filepath, lineno, docstring, signature
- `test_summary`: aggregated JUnit test results (optional)
- `profile_summary`: aggregated cProfile results (optional)
- `benchmark_summary`: aggregated benchmark results (optional)

## Usage

```bash
python -m falling_sand.indexer \\
  --source-root src \\
  --tests-root tests \\
  --output artifacts/index.json \\
  --exclude-dir .venv \\
  --test-report artifacts/unit-tests.xml \\
  --test-report artifacts/integration-tests.xml \\
  --profile-stats artifacts/profile.pstats \\
  --benchmark-report artifacts/benchmark.json
```

Use `--allow-missing-tests` to skip indexing when the tests root is absent.

You can pass `--exclude-dir` multiple times to add additional directories to ignore.
