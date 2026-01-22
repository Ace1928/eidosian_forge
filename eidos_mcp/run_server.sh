#!/bin/bash
export PATH="/home/lloyd/eidosian_venv/bin:$PATH"
export PYTHONPATH="/home/lloyd/eidosian_forge/eidos_mcp/src:$PYTHONPATH"
export EIDOS_FORGE_DIR="/home/lloyd/eidosian_forge"
exec /home/lloyd/eidosian_venv/bin/python3 -m eidos_mcp.eidos_mcp_server
