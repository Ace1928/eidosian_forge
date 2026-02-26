#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VENV_PYTHON="${FORGE_ROOT}/eidosian_venv/bin/python"
SERVICE_SCRIPT="${FORGE_ROOT}/doc_forge/scripts/scribe_service.py"
PORT_REGISTRY_SCRIPT="${FORGE_ROOT}/scripts/port_registry.py"

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[doc-forge] missing python: ${VENV_PYTHON}" >&2
  exit 1
fi

if [ ! -f "${SERVICE_SCRIPT}" ]; then
  echo "[doc-forge] missing service script: ${SERVICE_SCRIPT}" >&2
  exit 1
fi

export EIDOS_FORGE_ROOT="${FORGE_ROOT}"
DEFAULT_DOC_PORT="$("${VENV_PYTHON}" "${PORT_REGISTRY_SCRIPT}" get --service doc_forge_dashboard --field port --default 8930 2>/dev/null || echo 8930)"
DEFAULT_DOC_LLM_PORT="$("${VENV_PYTHON}" "${PORT_REGISTRY_SCRIPT}" get --service doc_forge_llm --field port --default 8093 2>/dev/null || echo 8093)"
MODEL_FROM_SELECTION="$("${VENV_PYTHON}" - <<'PY' 2>/dev/null
import json
from pathlib import Path
path = Path("config/model_selection.json")
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = ((((payload.get("services") or {}).get("doc_forge") or {}).get("completion_model") or "").strip())
        if model:
            print(model)
    except Exception:
        pass
PY
)"
MODEL_FROM_SWEEP="$("${VENV_PYTHON}" - <<'PY' 2>/dev/null
import json
from pathlib import Path
path = Path("reports/graphrag_sweep/model_selection_latest.json")
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = ((payload.get("winner") or {}).get("model_path") or "").strip()
        if model:
            print(model)
    except Exception:
        pass
PY
)"
export EIDOS_DOC_FORGE_PORT="${EIDOS_DOC_FORGE_PORT:-${DEFAULT_DOC_PORT}}"
export EIDOS_DOC_FORGE_HOST="${EIDOS_DOC_FORGE_HOST:-127.0.0.1}"
export EIDOS_DOC_FORGE_MODEL="${EIDOS_DOC_FORGE_MODEL:-${MODEL_FROM_SELECTION:-${MODEL_FROM_SWEEP:-models/Qwen2.5-0.5B-Instruct-Q8_0.gguf}}}"
export EIDOS_DOC_FORGE_SOURCE_ROOT="${EIDOS_DOC_FORGE_SOURCE_ROOT:-${FORGE_ROOT}}"
export EIDOS_DOC_FORGE_ENABLE_MANAGED_LLM="${EIDOS_DOC_FORGE_ENABLE_MANAGED_LLM:-1}"
export EIDOS_DOC_FORGE_LLM_PORT="${EIDOS_DOC_FORGE_LLM_PORT:-${DEFAULT_DOC_LLM_PORT}}"

cd "${FORGE_ROOT}"
exec "${VENV_PYTHON}" "${SERVICE_SCRIPT}" --host "${EIDOS_DOC_FORGE_HOST}" --port "${EIDOS_DOC_FORGE_PORT}"
