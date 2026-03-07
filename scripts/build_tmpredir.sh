#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SRC="${FORGE_ROOT}/lib/eidos_tmpredir/redirect_tmp.c"
OUT="${FORGE_ROOT}/build/libeidos_tmpredir.so"
CC_BIN="${CC:-}"

if [ -z "${CC_BIN}" ]; then
    if command -v clang >/dev/null 2>&1; then
        CC_BIN="$(command -v clang)"
    elif command -v gcc >/dev/null 2>&1; then
        CC_BIN="$(command -v gcc)"
    elif command -v cc >/dev/null 2>&1; then
        CC_BIN="$(command -v cc)"
    else
        echo "[tmpredir] missing C compiler" >&2
        exit 1
    fi
fi

mkdir -p "${FORGE_ROOT}/build"
"${CC_BIN}" -shared -fPIC -O2 -Wall -Wextra -o "${OUT}" "${SRC}" -ldl
printf '%s\n' "${OUT}"
