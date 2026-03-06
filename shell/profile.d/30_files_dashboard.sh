#!/usr/bin/env bash

eidos_termux_files_dashboard_init() {
    eidos_termux_is_interactive || return 0

    export FILES_DASHBOARD_DIR="${FILES_DASHBOARD_DIR:-/storage/emulated/0/Download/.files_dashboard}"
    export EIDOS_ENABLE_FILES_DASHBOARD_AUTOSTART="${EIDOS_ENABLE_FILES_DASHBOARD_AUTOSTART:-1}"

    alias filesdash='sh "$FILES_DASHBOARD_DIR/service.sh"'
    alias filesdash-start='sh "$FILES_DASHBOARD_DIR/service.sh" start'
    alias filesdash-stop='sh "$FILES_DASHBOARD_DIR/service.sh" stop'
    alias filesdash-status='sh "$FILES_DASHBOARD_DIR/service.sh" status'
    alias filesdash-url='sh "$FILES_DASHBOARD_DIR/service.sh" url'
    alias localhost:files='termux-open-url "http://files.localhost:8942"'

    if [ "${EIDOS_ENABLE_FILES_DASHBOARD_AUTOSTART}" = "1" ] && [ -f "$FILES_DASHBOARD_DIR/service.sh" ]; then
        sh "$FILES_DASHBOARD_DIR/service.sh" start >/dev/null 2>&1 || true
    fi
}
