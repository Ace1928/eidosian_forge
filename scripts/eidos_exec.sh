#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LIB_PATH="${FORGE_ROOT}/build/libeidos_tmpredir.so"

if [ -n "${EIDOS_TMPDIR:-}" ]; then
    target_tmp="${EIDOS_TMPDIR}"
elif [ -n "${PREFIX:-}" ] && printf '%s' "${PREFIX}" | grep -q 'com.termux'; then
    target_tmp="${PREFIX}/tmp/eidos-${USER:-termux}"
else
    target_tmp="${HOME}/tmp"
fi
mkdir -p "${target_tmp}"
export EIDOS_TMPDIR="${target_tmp}"
export TMPDIR="${target_tmp}"
export TMP="${target_tmp}"
export TEMP="${target_tmp}"

if [ -f "${LIB_PATH}" ] && [ "${EIDOS_ENABLE_TMP_PRELOAD:-0}" = "1" ]; then
    case ":${LD_PRELOAD:-}:" in
        *":${LIB_PATH}:"*) ;;
        *) export LD_PRELOAD="${LIB_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
    esac
fi

exec "$@"
