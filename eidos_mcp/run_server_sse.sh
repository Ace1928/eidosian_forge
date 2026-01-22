#!/bin/bash
export PATH="/home/lloyd/eidosian_forge/google-cloud-sdk/bin:/home/lloyd/eidosian_forge/eidosian_venv/bin:$PATH"
export PYTHONPATH="/home/lloyd/eidosian_forge/eidos_mcp/src:$PYTHONPATH"
export EIDOS_FORGE_DIR="/home/lloyd/eidosian_forge"
export EIDOS_MCP_TRANSPORT="sse"
export FASTMCP_PORT="8765"
export FASTMCP_HOST="127.0.0.1"
export PYTHONUNBUFFERED=1

# Run the server
exec /home/lloyd/eidosian_forge/eidosian_venv/bin/python3 -m eidos_mcp.eidos_mcp_server