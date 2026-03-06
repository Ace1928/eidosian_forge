#!/usr/bin/env bash

_eidos_capture_shell_baseline() {
    if [ -n "${EIDOS_SHELL_BASELINE_CAPTURED:-}" ]; then
        return 0
    fi
    export EIDOS_SHELL_BASELINE_CAPTURED=1
    export EIDOS_BASE_PATH="$(_eidos_clean_path "${PATH:-}")"
    export EIDOS_BASE_PYTHONPATH="$(_eidos_clean_pythonpath "${PYTHONPATH:-}")"
    export EIDOS_BASE_LD_LIBRARY_PATH="$(_eidos_clean_path "${LD_LIBRARY_PATH:-}")"
    case "${VIRTUAL_ENV:-}" in
        "$FORGE_ROOT"|"$FORGE_ROOT"/*) export EIDOS_BASE_VIRTUAL_ENV="" ;;
        *) export EIDOS_BASE_VIRTUAL_ENV="${VIRTUAL_ENV:-}" ;;
    esac
}

_eidos_should_activate() {
    case "${PWD:-}" in
        "$FORGE_ROOT"|"$FORGE_ROOT"/*) return 0 ;;
    esac
    [ "${EIDOS_AUTO_ACTIVATE_FORGE:-0}" = "1" ]
}
