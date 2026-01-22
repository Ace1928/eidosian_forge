# Testing and Profiling

## Full verification
Run the full verification suite after any change:

```bash
scripts/verify.sh
```

The script prefers `.venv/bin/python` if available.

## Indexing options
Use `--exclude-dir` and `--allow-missing-tests` to control indexing behavior when needed.

## Unit tests

```bash
python -m pytest tests/unit
```

## Integration tests

```bash
python -m pytest tests/integration -m integration
```

## Profiling

```bash
python scripts/profile_index.py
```

## Benchmarking

```bash
python scripts/benchmark_suite.py
```

## Linting and typing (optional)

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src
```
