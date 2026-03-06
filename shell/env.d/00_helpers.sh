#!/usr/bin/env bash

_eidos_norm_path() {
    if [ -n "${1:-}" ] && [ -d "$1" ]; then
        (cd "$1" 2>/dev/null && pwd -P) || printf '%s\n' "$1"
    else
        printf '%s\n' "${1:-}"
    fi
}

_eidos_path_contains() {
    case ":${1:-}:" in
        *":${2:-}:"*) return 0 ;;
        *) return 1 ;;
    esac
}

_eidos_path_prepend_unique() {
    local current="${1:-}" entry="${2:-}"
    [ -n "$entry" ] || { printf '%s\n' "$current"; return 0; }
    if _eidos_path_contains "$current" "$entry"; then
        printf '%s\n' "$current"
    elif [ -n "$current" ]; then
        printf '%s:%s\n' "$entry" "$current"
    else
        printf '%s\n' "$entry"
    fi
}

_eidos_clean_path() {
    local raw="${1:-}" result="" entry="" normalized=""
    local IFS=':'
    for entry in $raw; do
        [ -n "$entry" ] || continue
        normalized="$(_eidos_norm_path "$entry")"
        case "$normalized" in
            "$FORGE_ROOT"|"$FORGE_ROOT"/*) continue ;;
            "$HOME/.codex/tmp"|"$HOME/.codex/tmp"/*) continue ;;
        esac
        if ! _eidos_path_contains "$result" "$normalized"; then
            if [ -n "$result" ]; then
                result="${result}:$normalized"
            else
                result="$normalized"
            fi
        fi
    done
    printf '%s\n' "$result"
}

_eidos_clean_pythonpath() {
    local raw="${1:-}" result="" entry="" normalized=""
    local IFS=':'
    for entry in $raw; do
        [ -n "$entry" ] || continue
        normalized="$(_eidos_norm_path "$entry")"
        case "$normalized" in
            "$FORGE_ROOT"|"$FORGE_ROOT"/*) continue ;;
        esac
        if ! _eidos_path_contains "$result" "$normalized"; then
            if [ -n "$result" ]; then
                result="${result}:$normalized"
            else
                result="$normalized"
            fi
        fi
    done
    printf '%s\n' "$result"
}
