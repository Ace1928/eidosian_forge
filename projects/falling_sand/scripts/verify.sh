#!/usr/bin/env bash
set -euo pipefail

PYTHON="python"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest tests/unit --junitxml artifacts/unit-tests.xml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest tests/integration -m integration --junitxml artifacts/integration-tests.xml
"$PYTHON" scripts/profile_index.py
"$PYTHON" scripts/benchmark_suite.py
"$PYTHON" -m falling_sand.indexer \
  --output artifacts/index.json \
  --test-report artifacts/unit-tests.xml \
  --test-report artifacts/integration-tests.xml \
  --profile-stats artifacts/profile.pstats \
  --benchmark-report artifacts/benchmark.json
"$PYTHON" scripts/ingest_index.py --index artifacts/index.json --db artifacts/index.db
"$PYTHON" scripts/report_trends.py --db artifacts/index.db --output artifacts/report.json

if "$PYTHON" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("ruff") else 1)
PY
then
  "$PYTHON" -m ruff check src tests scripts
else
  echo "ruff not installed; skipping lint" >&2
fi

if "$PYTHON" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("mypy") else 1)
PY
then
  "$PYTHON" -m mypy src
else
  echo "mypy not installed; skipping type check" >&2
fi
