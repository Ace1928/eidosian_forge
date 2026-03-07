#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FORGE_ROOT}/eidosian_venv/bin/python"

INTERVAL_SEC="${EIDOS_SCHEDULER_INTERVAL_SEC:-900}"
TIMEOUT_SEC="${EIDOS_SCHEDULER_TIMEOUT_SEC:-7200}"
MAX_CYCLES="${EIDOS_SCHEDULER_MAX_CYCLES:-0}"
MODEL="${EIDOS_QWEN_MODEL:-qwen3.5:2b}"
CODE_MAX_FILES="${EIDOS_SCHEDULER_CODE_MAX_FILES:-}"
RUN_GRAPHRAG="${EIDOS_SCHEDULER_RUN_GRAPHRAG:-1}"

cmd=(
  "${PYTHON_BIN}" "${FORGE_ROOT}/scripts/eidos_scheduler.py"
  --interval-sec "${INTERVAL_SEC}"
  --timeout-sec "${TIMEOUT_SEC}"
  --max-cycles "${MAX_CYCLES}"
  --model "${MODEL}"
)

if [[ -n "${CODE_MAX_FILES}" ]]; then
  cmd+=(--code-max-files "${CODE_MAX_FILES}")
fi
if [[ "${RUN_GRAPHRAG}" == "1" || "${RUN_GRAPHRAG}" == "true" || "${RUN_GRAPHRAG}" == "yes" ]]; then
  cmd+=(--run-graphrag)
fi

exec "${cmd[@]}"
