#!/usr/bin/env bash

EIDOS_TERMUX_PROFILE_DIR="${EIDOS_TERMUX_PROFILE_DIR:-/data/data/com.termux/files/home/eidosian_forge/shell/profile.d}"

for eidos_termux_module in \
    "${EIDOS_TERMUX_PROFILE_DIR}/00_termux_helpers.sh" \
    "${EIDOS_TERMUX_PROFILE_DIR}/10_termux_runtime.sh" \
    "${EIDOS_TERMUX_PROFILE_DIR}/20_eidos_bootstrap.sh" \
    "${EIDOS_TERMUX_PROFILE_DIR}/30_files_dashboard.sh" \
    "${EIDOS_TERMUX_PROFILE_DIR}/40_npm_completion.sh"
do
    [ -f "${eidos_termux_module}" ] || continue
    # shellcheck source=/dev/null
    source "${eidos_termux_module}"
done

unset eidos_termux_module

eidos_termux_runtime_init
eidos_termux_bootstrap_init
eidos_termux_files_dashboard_init
eidos_termux_npm_completion_init
