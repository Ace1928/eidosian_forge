#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="/data/data/com.termux/files/home/eidosian_forge"
EIDOS_ENV_MODULE="$FORGE_ROOT/eidos_env.sh"

if [ ! -f "$EIDOS_ENV_MODULE" ]; then
    printf 'Missing environment module: %s\n' "$EIDOS_ENV_MODULE" >&2
    exit 2
fi

# shellcheck source=/dev/null
source "$EIDOS_ENV_MODULE"

if [ "$#" -eq 0 ]; then
    printf 'Usage: %s <command> [args...]\n' "$0" >&2
    exit 2
fi

eidos_safe_exec "$@"
