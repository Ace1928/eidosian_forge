#!/bin/bash
set -euo pipefail

# Eidosian MCP Server launcher (Streamable HTTP transport)
# Serves on http://127.0.0.1:8928/mcp by default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_BIN="${FORGE_ROOT}/eidosian_venv/bin"
PYTHON_BIN="${VENV_BIN}/python3"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Missing venv python at ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${VENV_BIN}:$PATH"
export EIDOS_FORGE_DIR="${FORGE_ROOT}"
export EIDOS_MCP_TRANSPORT="${EIDOS_MCP_TRANSPORT:-streamable-http}"
export EIDOS_MCP_MOUNT_PATH="${EIDOS_MCP_MOUNT_PATH:-/mcp}"
export FASTMCP_PORT="${FASTMCP_PORT:-8928}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export FASTMCP_RELOAD="${FASTMCP_RELOAD:-false}"
export EIDOS_MCP_ENABLE_COMPAT_HEADERS="${EIDOS_MCP_ENABLE_COMPAT_HEADERS:-1}"
export EIDOS_MCP_ENABLE_SESSION_RECOVERY="${EIDOS_MCP_ENABLE_SESSION_RECOVERY:-1}"
export EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT="${EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT:-1}"
export EIDOS_MCP_ENFORCE_ORIGIN="${EIDOS_MCP_ENFORCE_ORIGIN:-1}"
export EIDOS_MCP_ALLOWED_ORIGINS="${EIDOS_MCP_ALLOWED_ORIGINS:-http://127.0.0.1,http://localhost,http://[::1]}"
export PYTHONUNBUFFERED=1

# Avoid UI notifications when this script runs as a background service.
export EIDOS_DISABLE_NOTIFICATIONS=1

# Ensure module discovery for src-layout packages.
source "${FORGE_ROOT}/eidos_env.sh"
export PYTHONPATH="${SCRIPT_DIR}/src:${FORGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure local core package is importable from this checkout.
if ! "${PYTHON_BIN}" -c "import eidosian_core" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -e "${FORGE_ROOT}/lib" >/dev/null
fi

exec "${PYTHON_BIN}" -m eidos_mcp.eidos_mcp_server
