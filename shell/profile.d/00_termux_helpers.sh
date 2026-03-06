#!/usr/bin/env bash

eidos_termux_is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [ -n "${PREFIX:-}" ] && case "${PREFIX:-}" in
        *com.termux*) return 0 ;;
    esac
    return 1
}

eidos_termux_is_interactive() {
    case "$-" in
        *i*) return 0 ;;
        *) return 1 ;;
    esac
}

eidos_termux_log() {
    local level="${1:-INFO}"
    shift || true
    if command -v log_info >/dev/null 2>&1 && [ "${level}" = "INFO" ]; then
        log_info "$*"
        return 0
    fi
    if command -v log_warn >/dev/null 2>&1 && [ "${level}" = "WARN" ]; then
        log_warn "$*"
        return 0
    fi
    if command -v log_error >/dev/null 2>&1 && [ "${level}" = "ERROR" ]; then
        log_error "$*"
        return 0
    fi
    printf '[%s] %s\n' "${level}" "$*"
}
