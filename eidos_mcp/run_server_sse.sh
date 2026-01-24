#!/bin/bash
VENV_BIN="/home/lloyd/eidosian_forge/eidosian_venv/bin"
export PATH="/home/lloyd/eidosian_forge/google-cloud-sdk/bin:${VENV_BIN}:$PATH"
export PYTHONPATH="/home/lloyd/eidosian_forge/eidos_mcp/src:$PYTHONPATH"
export EIDOS_FORGE_DIR="/home/lloyd/eidosian_forge"
export EIDOS_MCP_TRANSPORT="sse"
export FASTMCP_PORT="${FASTMCP_PORT:-8928}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export PYTHONUNBUFFERED=1
exec "${VENV_BIN}/python3" -m eidos_mcp.eidos_mcp_server
