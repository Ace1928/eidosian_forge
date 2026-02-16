#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/eidosian_venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/eidosian_venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="$ROOT_DIR/lib:$ROOT_DIR/agent_forge/src:$ROOT_DIR/eidos_mcp/src:$ROOT_DIR/crawl_forge/src"
TMP_BASE="${TMPDIR:-/tmp}"
if [[ ! -w "$TMP_BASE" ]]; then
  TMP_BASE="/data/data/com.termux/files/usr/tmp"
fi
mkdir -p "$TMP_BASE"

echo "== Linux Parity Smoke =="
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "-- pytest matrix --"
"$PYTHON_BIN" -m pytest -q \
  agent_forge/tests/test_consciousness_milestone_a.py \
  agent_forge/tests/test_consciousness_milestone_b.py \
  agent_forge/tests/test_consciousness_milestone_d.py \
  agent_forge/tests/test_consciousness_trials.py \
  agent_forge/tests/test_consciousness_benchmarks.py \
  agent_forge/tests/test_consciousness_integrated_benchmark.py \
  agent_forge/tests/test_workspace.py \
  agent_forge/tests/test_db_and_daemon.py \
  crawl_forge/tests/test_tika_extractor_fallback.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  eidos_mcp/tests/test_mcp_tool_calls_individual.py \
  eidos_mcp/tests/test_diagnostics_transport_matrix.py

echo "-- stdio integration --"
EIDOS_RUN_FULL_INTEGRATION=1 "$PYTHON_BIN" -m pytest -q eidos_mcp/tests/test_mcp_tools_stdio.py

echo "-- protocol + runtime audit --"
"$PYTHON_BIN" scripts/run_consciousness_protocol.py --trials 2 > "$TMP_BASE/consciousness_protocol_smoke.json"
"$PYTHON_BIN" scripts/audit_mcp_tools_resources.py --timeout 8

echo "Linux parity smoke PASS"
