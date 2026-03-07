#!/usr/bin/env bash

eidos_termux_is_termux() {
    eidos_shell_is_termux
}

eidos_termux_is_interactive() {
    eidos_shell_is_interactive
}

eidos_termux_log() {
    local level="${1:-INFO}"
    shift || true
    eidos_shell_log "${level}" "$*"
}
