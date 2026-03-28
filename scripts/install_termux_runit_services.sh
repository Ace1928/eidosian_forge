#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-${HOME}/eidosian_forge}"
PREFIX_ROOT="${PREFIX:-/data/data/com.termux/files/usr}"
SERVICE_ROOT="${EIDOS_RUNIT_SERVICE_DIR:-${PREFIX_ROOT}/var/service}"
LOG_ROOT="${EIDOS_RUNIT_LOG_ROOT:-${HOME}/.eidosian/log/sv}"
TERMUX_SVLOGGER="${PREFIX_ROOT}/share/termux-services/svlogger"
BASH_BIN="${PREFIX_ROOT}/bin/bash"
SH_BIN="${PREFIX_ROOT}/bin/sh"

mkdir -p "${SERVICE_ROOT}" "${LOG_ROOT}"

if [ ! -x "${BASH_BIN}" ]; then
    BASH_BIN="$(command -v bash || true)"
fi
if [ ! -x "${SH_BIN}" ]; then
    SH_BIN="$(command -v sh || true)"
fi
if [ -z "${BASH_BIN}" ] || [ -z "${SH_BIN}" ]; then
    echo "missing shell runtime for runit service generation" >&2
    exit 1
fi

_write_run_script() {
    local service_dir="$1"
    local entry_path="$2"
    local extra_pythonpath="${3:-}"
    cat > "${service_dir}/run" <<EOF
#!${BASH_BIN}
set -euo pipefail
export EIDOS_FORGE_ROOT="${FORGE_ROOT}"
export EIDOS_SERVICE_SUPERVISION="runit"
EOF
    if [ -n "${extra_pythonpath}" ]; then
        cat >> "${service_dir}/run" <<EOF
export PYTHONPATH="${extra_pythonpath}:\${PYTHONPATH:-}"
EOF
    fi
    cat >> "${service_dir}/run" <<EOF
cd "${FORGE_ROOT}"
exec "${entry_path}"
EOF
    chmod 0755 "${service_dir}/run"
}

_write_log_script() {
    local service_dir="$1"
    local service_name="$2"
    mkdir -p "${service_dir}/log"
    if [ -x "${TERMUX_SVLOGGER}" ]; then
        ln -sfn "${TERMUX_SVLOGGER}" "${service_dir}/log/run"
        return 0
    fi
    cat > "${service_dir}/log/run" <<EOF
#!${SH_BIN}
mkdir -p "${LOG_ROOT}/${service_name}"
if command -v svlogd >/dev/null 2>&1; then
    exec svlogd -tt "${LOG_ROOT}/${service_name}"
fi
exec cat
EOF
    chmod 0755 "${service_dir}/log/run"
}

_install_service() {
    local service_name="$1"
    local entry_path="$2"
    local extra_pythonpath="${3:-}"
    local service_dir="${SERVICE_ROOT}/${service_name}"

    mkdir -p "${service_dir}"
    _write_run_script "${service_dir}" "${entry_path}" "${extra_pythonpath}"
    _write_log_script "${service_dir}" "${service_name}"
    : > "${service_dir}/down"
}

_install_service "eidos-ollama-qwen" "${FORGE_ROOT}/scripts/run_ollama_qwen.sh"
_install_service "eidos-ollama-embedding" "${FORGE_ROOT}/scripts/run_ollama_embedding.sh"
_install_service "eidos-mcp" "${FORGE_ROOT}/eidos_mcp/run_server.sh"
_install_service "eidos-doc-forge" "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh"
_install_service "eidos-atlas" "${FORGE_ROOT}/atlas_forge/scripts/run_atlas.sh" "${FORGE_ROOT}/atlas_forge/src:${FORGE_ROOT}/web_interface_forge/src:${FORGE_ROOT}/memory_forge/src:${FORGE_ROOT}/knowledge_forge/src:${FORGE_ROOT}/word_forge/src:${FORGE_ROOT}/lib"
_install_service "eidos-scheduler" "${FORGE_ROOT}/scripts/run_eidos_scheduler.sh"
_install_service "eidos-local-agent" "${FORGE_ROOT}/scripts/run_local_mcp_agent.sh"

printf '%s\n' "${SERVICE_ROOT}"
