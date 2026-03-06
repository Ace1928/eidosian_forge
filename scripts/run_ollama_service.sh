#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${FORGE_ROOT}/eidosian_venv/bin/python"
PORT_REGISTRY_SCRIPT="${FORGE_ROOT}/scripts/port_registry.py"

ROLE="${1:-qwen}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[ollama-service] missing ollama binary" >&2
  exit 1
fi

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "[ollama-service] missing python: ${VENV_PYTHON}" >&2
  exit 1
fi

case "${ROLE}" in
  qwen)
    SERVICE_KEY="ollama_qwen_http"
    DEFAULT_PORT="8938"
    MODEL="${EIDOS_QWEN_MODEL:-qwen3.5:2b}"
    WARM_ENDPOINT="/api/generate"
    WARM_PAYLOAD_TEMPLATE='{"model":"%s","prompt":"Reply with READY only.","stream":false,"keep_alive":"24h","options":{"num_predict":8,"temperature":0}}'
    ;;
  embedding|embed)
    SERVICE_KEY="ollama_embedding_http"
    DEFAULT_PORT="8940"
    MODEL="${EIDOS_EMBED_MODEL:-nomic-embed-text}"
    WARM_ENDPOINT="/api/embeddings"
    WARM_PAYLOAD_TEMPLATE='{"model":"%s","prompt":"warmup"}'
    ;;
  *)
    echo "[ollama-service] unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

HOST="$("${VENV_PYTHON}" "${PORT_REGISTRY_SCRIPT}" get --service "${SERVICE_KEY}" --field host --default 127.0.0.1 2>/dev/null || echo 127.0.0.1)"
PORT="$("${VENV_PYTHON}" "${PORT_REGISTRY_SCRIPT}" get --service "${SERVICE_KEY}" --field port --default "${DEFAULT_PORT}" 2>/dev/null || echo "${DEFAULT_PORT}")"
BASE_URL="http://${HOST}:${PORT}"

export OLLAMA_HOST="${BASE_URL}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup INT TERM EXIT

echo "[ollama-service] starting role=${ROLE} model=${MODEL} url=${BASE_URL}"
ollama serve &
SERVER_PID=$!

wait_http() {
  local waited=0
  while [ "${waited}" -lt 60 ]; do
    if curl -fsS --max-time 2 "${BASE_URL}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    waited=$((waited + 1))
    sleep 1
  done
  return 1
}

if ! wait_http; then
  echo "[ollama-service] ${ROLE} server failed to become healthy at ${BASE_URL}" >&2
  exit 1
fi

warm_payload="$(printf "${WARM_PAYLOAD_TEMPLATE}" "${MODEL}")"
if ! curl -fsS --max-time 180 -H 'Content-Type: application/json' \
  -d "${warm_payload}" "${BASE_URL}${WARM_ENDPOINT}" >/dev/null 2>&1; then
  echo "[ollama-service] warning: warmup failed for role=${ROLE} model=${MODEL} at ${BASE_URL}" >&2
else
  echo "[ollama-service] warmed role=${ROLE} model=${MODEL}"
fi

wait "${SERVER_PID}"
