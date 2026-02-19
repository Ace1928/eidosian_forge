#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/eidosian_venv/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" scripts/download_local_models.py --profile graphrag

echo "\nGraphRAG model inventory:"
ls -lh models/ | sed -n '1,200p'
