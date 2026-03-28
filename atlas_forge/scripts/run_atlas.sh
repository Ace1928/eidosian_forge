#!/bin/bash
# run_atlas.sh - Start the Eidosian Atlas Dashboard

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
PYTHON_BIN="${FORGE_ROOT}/eidosian_venv/bin/python"
PORT="${EIDOS_DASHBOARD_PORT:-8936}"

echo "[atlas] Starting Atlas Forge Dashboard on port ${PORT}..."

# Ensure we are in the forge root for path resolution
cd "${FORGE_ROOT}"

# Setup PYTHONPATH to include all forges
export PYTHONPATH="${FORGE_ROOT}/atlas_forge/src:${FORGE_ROOT}/web_interface_forge/src:${FORGE_ROOT}/memory_forge/src:${FORGE_ROOT}/knowledge_forge/src:${FORGE_ROOT}/word_forge/src:${FORGE_ROOT}/lib:${PYTHONPATH}"

# Start the dashboard using uvicorn
"${PYTHON_BIN}" -m uvicorn atlas_forge.app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info
