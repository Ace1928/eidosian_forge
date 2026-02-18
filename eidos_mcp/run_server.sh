#!/bin/bash
# Eidosian MCP Server launcher (Streamable HTTP transport)
# Serves on http://localhost:8928/mcp

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_BIN="${FORGE_ROOT}/eidosian_venv/bin"

export PATH="${VENV_BIN}:$PATH"
export EIDOS_FORGE_DIR="${FORGE_ROOT}"
export EIDOS_MCP_TRANSPORT="streamable-http"
export EIDOS_MCP_MOUNT_PATH="/mcp"
export FASTMCP_PORT="${FASTMCP_PORT:-8928}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export FASTMCP_RELOAD="${FASTMCP_RELOAD:-true}"
export PYTHONUNBUFFERED=1

# Ensure uvicorn can find the eidos_mcp package and other forge modules
source "${FORGE_ROOT}/eidos_env.sh"
export PYTHONPATH="${SCRIPT_DIR}/src:${FORGE_ROOT}:${PYTHONPATH}"

# Ensure local core package is installed from source checkout.
if ! "${VENV_BIN}/python3" -c "import eidosian_core" >/dev/null 2>&1; then
  "${VENV_BIN}/python3" -m pip install -e "${FORGE_ROOT}/lib" >/dev/null
fi

exec "${VENV_BIN}/python3" -m eidos_mcp.eidos_mcp_server
