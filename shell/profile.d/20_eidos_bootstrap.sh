#!/usr/bin/env bash

eidos_termux_bootstrap_init() {
    eidos_termux_is_interactive || return 0

    export EIDOS_FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
    export EIDOS_AUTO_ACTIVATE_FORGE="${EIDOS_AUTO_ACTIVATE_FORGE:-1}"
    export EIDOS_ENABLE_DOC_FORGE_AUTOSTART="${EIDOS_ENABLE_DOC_FORGE_AUTOSTART:-1}"
    export EIDOS_DOC_FORGE_PORT="${EIDOS_DOC_FORGE_PORT:-8930}"

    local env_module="${EIDOS_FORGE_ROOT}/eidos_env.sh"
    local service_script="${EIDOS_FORGE_ROOT}/scripts/eidos_termux_services.sh"

    if [ -f "${env_module}" ]; then
        # shellcheck source=/dev/null
        source "${env_module}"
    else
        eidos_termux_log WARN "Failed to find Eidosian Environment Module at ${env_module}"
    fi

    if [ -x "${service_script}" ]; then
        "${service_script}" start-shell >/dev/null 2>&1 || eidos_termux_log WARN "Eidos service bootstrap failed."
        trap "${service_script} exit-shell >/dev/null 2>&1 || true" EXIT
    else
        eidos_termux_log WARN "Eidos service manager missing at ${service_script}"
    fi

    if [ "${EIDOS_DISABLE_NOTIFICATIONS:-0}" != "1" ] && command -v notify >/dev/null 2>&1; then
        notify "💎 Eidosian Nexus initialized. Status: PERFECT."
    fi
}
