#!/bin/bash
# run_dashboard.sh - LEGACY WRAPPER
# This script is now a legacy wrapper for the new Atlas Forge dashboard.

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
NEW_SCRIPT="${FORGE_ROOT}/atlas_forge/scripts/run_atlas.sh"

if [ -x "${NEW_SCRIPT}" ]; then
    echo "[legacy] Delegating to new Atlas Forge script..."
    exec "${NEW_SCRIPT}" "$@"
else
    echo "[error] New Atlas Forge script not found at ${NEW_SCRIPT}"
    exit 1
fi
