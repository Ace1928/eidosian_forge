#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TMP_DIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}"
STATUS_JSON="${TMP_DIR}/eidos_status_$$.json"
PROBLEMS=0

cleanup() {
  rm -f "${STATUS_JSON}" 2>/dev/null || true
}
trap cleanup EXIT

echo "== Eidosian Termux Audit =="
echo "Forge root: ${FORGE_ROOT}"
echo "Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo

echo "-- Runtime --"
echo "uname: $(uname -a)"
echo "shell: ${SHELL:-unknown}"
echo "prefix: ${PREFIX:-unset}"
echo "home: ${HOME:-unset}"
echo

echo "-- Toolchain --"
for cmd in python pip node npm clang git cmake make rustc cargo go java rg; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    version="$("${cmd}" --version 2>/dev/null | head -n 1 || true)"
    echo "${cmd}: ${version:-installed}"
  else
    echo "${cmd}: missing"
  fi
done
echo

echo "-- Forge Status --"
if timeout 180 python "${FORGE_ROOT}/bin/eidosian" --json status >"${STATUS_JSON}"; then
  if ! python - "${STATUS_JSON}" <<'PY'; then
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
data = json.loads(status_path.read_text())
forges = data.get("forges", {})
bad = {
    name: info for name, info in forges.items()
    if (not info.get("available")) or info.get("status") == "error"
}
print(f"mcp_tools: {data.get('mcp_tools', 0)}")
print(f"memories: {data.get('memories', 0)}")
print(f"knowledge_nodes: {data.get('knowledge_nodes', 0)}")
print(f"code_elements: {data.get('code_elements', 0)}")
print(f"problem_forges: {len(bad)}")
for name, info in sorted(bad.items()):
    print(f"  - {name}: {info.get('error', info.get('status', 'unknown'))}")
sys.exit(1 if bad else 0)
PY
    PROBLEMS=$((PROBLEMS + 1))
  fi
else
  echo "failed: eidosian status command"
  PROBLEMS=$((PROBLEMS + 1))
fi
echo

echo "-- MCP Persona Fetch --"
if timeout 90 python "${FORGE_ROOT}/eidos_mcp/eidos_fetch.py" "eidos://persona" >/dev/null; then
  echo "ok: eidos_fetch persona"
else
  echo "failed: eidos_fetch persona"
  PROBLEMS=$((PROBLEMS + 1))
fi
echo

echo "-- Venv Health --"
VENV_DIR="${FORGE_ROOT}/eidosian_venv"
if [[ -d "${VENV_DIR}" ]]; then
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "ok: ${VENV_DIR}/bin/python"
  else
    echo "warn: ${VENV_DIR}/bin/python is missing or not executable"
    echo "      run: ${FORGE_ROOT}/scripts/rebuild_eidosian_venv.sh --force"
  fi
  if [[ -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    echo "ok: pyvenv.cfg present"
  else
    echo "warn: pyvenv.cfg missing"
    echo "      run: ${FORGE_ROOT}/scripts/rebuild_eidosian_venv.sh --force"
  fi
else
  echo "warn: venv directory missing (${VENV_DIR})"
  echo "      run: ${FORGE_ROOT}/scripts/rebuild_eidosian_venv.sh --force"
fi
echo

echo "-- Hardcoded Path Scan --"
if command -v rg >/dev/null 2>&1; then
  if rg -n "/home/lloyd" \
      "${FORGE_ROOT}/bin" \
      "${FORGE_ROOT}/eidos_mcp/src/eidos_mcp" \
      "${FORGE_ROOT}/scripts/launch_codex_agent.sh" \
      "${FORGE_ROOT}/scripts/verify_mcp_health.py" \
      "${FORGE_ROOT}/scripts/sync_forges.py" \
      "${FORGE_ROOT}/scripts/bootstrap_foundations.py" \
      "${FORGE_ROOT}/scripts/audit_ports.py" \
      "${FORGE_ROOT}/scripts/codex_query.py" \
      "${FORGE_ROOT}/scripts/context_index.py" \
      "${FORGE_ROOT}/scripts/assess_migration.py" \
      "${FORGE_ROOT}/memory_forge/src" \
      "${FORGE_ROOT}/knowledge_forge/src" \
      "${FORGE_ROOT}/code_forge/src" \
      --glob '*.py' \
      --glob '*.sh' \
      --glob '*.bash' \
      --glob '!**/__pycache__/**' \
      --glob '!**/archive_forge/**' \
      --glob '!scripts/termux_audit.sh' \
      --glob '!scripts/AGENTS.md' \
      >/dev/null; then
    echo "warn: hardcoded /home/lloyd references found in active modules"
    PROBLEMS=$((PROBLEMS + 1))
  else
    echo "ok: no hardcoded /home/lloyd in active modules"
  fi
else
  echo "skip: rg not available"
fi
echo

if [[ "${PROBLEMS}" -eq 0 ]]; then
  echo "Audit result: PASS"
  exit 0
fi

echo "Audit result: FAIL (${PROBLEMS} issue(s))"
exit 1
