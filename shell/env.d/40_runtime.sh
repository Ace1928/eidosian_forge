#!/usr/bin/env bash

_eidos_capture_shell_baseline
if _eidos_should_activate; then
    eidos_use_env
else
    eidos_reset_env
fi

if [ -z "${EIDOS_DISABLE_NOTIFICATIONS:-}" ] && [ -n "${PS1:-}" ] && command -v termux-notification >/dev/null 2>&1; then
    termux-notification --title "Eidosian Nexus" --content "Environment module ready" --id 100 --priority low
fi
