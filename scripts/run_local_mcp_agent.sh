#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
PYTHON_BIN="${EIDOS_VENV_PYTHON:-${FORGE_ROOT}/eidosian_venv/bin/python}"

PROFILE="${EIDOS_LOCAL_AGENT_PROFILE:-observer}"
OBJECTIVE="${EIDOS_LOCAL_AGENT_OBJECTIVE:-Inspect forge health, use read-only MCP tools under policy, and report the single highest-leverage next action.}"
INTERVAL_SEC="${EIDOS_LOCAL_AGENT_INTERVAL_SEC:-180}"
MAX_CYCLES="${EIDOS_LOCAL_AGENT_MAX_CYCLES:-0}"
TIMEOUT_SEC="${EIDOS_LOCAL_AGENT_TIMEOUT_SEC:-1800}"
MODEL="${EIDOS_LOCAL_AGENT_MODEL:-qwen3.5:2b}"
POLICY_PATH="${EIDOS_LOCAL_AGENT_POLICY_PATH:-${FORGE_ROOT}/cfg/local_agent_profiles.json}"

cd "${FORGE_ROOT}"

exec "${PYTHON_BIN}" "${FORGE_ROOT}/scripts/eidos_local_agent.py" \
  "${OBJECTIVE}" \
  --profile "${PROFILE}" \
  --policy-path "${POLICY_PATH}" \
  --model "${MODEL}" \
  --continuous \
  --interval-sec "${INTERVAL_SEC}" \
  --max-cycles "${MAX_CYCLES}" \
  --timeout-sec "${TIMEOUT_SEC}"
