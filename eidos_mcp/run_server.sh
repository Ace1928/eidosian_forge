#!/bin/bash
# Eidosian MCP Server launcher (Streamable HTTP transport)
# Serves on http://localhost:8928/mcp

VENV_BIN="/home/lloyd/eidosian_forge/eidosian_venv/bin"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PATH="/home/lloyd/eidosian_forge/google-cloud-sdk/bin:${VENV_BIN}:$PATH"
export EIDOS_FORGE_DIR="/home/lloyd/eidosian_forge"
export EIDOS_MCP_TRANSPORT="streamable-http"
export EIDOS_MCP_MOUNT_PATH="/mcp"
export EIDOS_MCP_STATELESS_HTTP="${EIDOS_MCP_STATELESS_HTTP:-1}"
export EIDOS_MCP_ENABLE_COMPAT_HEADERS="${EIDOS_MCP_ENABLE_COMPAT_HEADERS:-1}"
export EIDOS_MCP_ENABLE_SESSION_RECOVERY="${EIDOS_MCP_ENABLE_SESSION_RECOVERY:-1}"
export EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT="${EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT:-1}"
export FASTMCP_PORT="${FASTMCP_PORT:-8928}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export PYTHONUNBUFFERED=1

# Ensure local core package is installed from source checkout.
if ! "${VENV_BIN}/python3" -c "import eidosian_core" >/dev/null 2>&1; then
  "${VENV_BIN}/python3" -m pip install -e "${FORGE_ROOT}/lib" >/dev/null
fi

exec "${VENV_BIN}/python3" -m eidos_mcp.eidos_mcp_server
