#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VENV_PYTHON="${FORGE_ROOT}/eidosian_venv/bin/python"
DASHBOARD_MODULE="web_interface_forge.dashboard.main:app"
PORT_REGISTRY_SCRIPT="${FORGE_ROOT}/scripts/port_registry.py"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[dashboard] missing python: ${VENV_PYTHON}" >&2
  exit 1
fi

export EIDOS_FORGE_ROOT="${FORGE_ROOT}"
DEFAULT_PORT="$("${VENV_PYTHON}" "${PORT_REGISTRY_SCRIPT}" get --service eidos_atlas_dashboard --field port --default 8936 2>/dev/null || echo 8936)"
export EIDOS_DASHBOARD_PORT="${EIDOS_DASHBOARD_PORT:-${DEFAULT_PORT}}"

# Ensure PYTHONPATH includes the src directory so the module can be found
export PYTHONPATH="${FORGE_ROOT}/web_interface_forge/src:${PYTHONPATH:-}"

echo "[dashboard] Starting Eidosian Atlas on port ${EIDOS_DASHBOARD_PORT}..."
exec "${VENV_PYTHON}" -m uvicorn "${DASHBOARD_MODULE}" --host 0.0.0.0 --port "${EIDOS_DASHBOARD_PORT}" --log-level info
