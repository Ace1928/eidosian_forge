#!/bin/bash
# Eidosian MCP Server launcher (Streamable HTTP transport)
# Serves on http://localhost:8928/mcp

VENV_BIN="/home/lloyd/eidosian_forge/eidosian_venv/bin"
export PATH="/home/lloyd/eidosian_forge/google-cloud-sdk/bin:${VENV_BIN}:$PATH"
export EIDOS_FORGE_DIR="/home/lloyd/eidosian_forge"
export EIDOS_MCP_TRANSPORT="streamable-http"
export EIDOS_MCP_MOUNT_PATH="/mcp"
export FASTMCP_PORT="${FASTMCP_PORT:-8928}"
export FASTMCP_HOST="${FASTMCP_HOST:-127.0.0.1}"
export PYTHONUNBUFFERED=1

exec "${VENV_BIN}/python3" -m eidos_mcp.eidos_mcp_server
