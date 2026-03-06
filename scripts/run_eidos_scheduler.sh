#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${FORGE_ROOT}/eidosian_venv/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[scheduler] missing python: ${VENV_PYTHON}" >&2
  exit 1
fi

export EIDOS_FORGE_ROOT="${FORGE_ROOT}"
export EIDOS_FORGE_DIR="${FORGE_ROOT}"
export PYTHONPATH="${FORGE_ROOT}/lib:${FORGE_ROOT}/code_forge/src:${FORGE_ROOT}/knowledge_forge/src:${FORGE_ROOT}/memory_forge/src:${FORGE_ROOT}/eidos_mcp/src:${FORGE_ROOT}/ollama_forge/src:${FORGE_ROOT}/web_interface_forge/src:${PYTHONPATH:-}"

exec "${VENV_PYTHON}" "${FORGE_ROOT}/scripts/eidos_scheduler.py" --run-graphrag "$@"
