#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
PYTHON_BIN="${EIDOS_VENV_PYTHON:-${FORGE_ROOT}/eidosian_venv/bin/python}"
PORT_REGISTRY_SCRIPT="${FORGE_ROOT}/scripts/port_registry.py"
START_SERVICES_SH="${PREFIX:-/data/data/com.termux/files/usr}/etc/profile.d/start-services.sh"
RUN_DIR="${HOME}/.eidosian/run"
LOCK_FILE="${RUN_DIR}/services.lock"
SHELL_COUNT_FILE="${RUN_DIR}/.interactive_shell_count"
MCP_PID_FILE="${RUN_DIR}/eidos_mcp.pid"
DOC_PID_FILE="${RUN_DIR}/doc_forge.pid"
ATLAS_PID_FILE="${RUN_DIR}/eidos_atlas.pid"
SCHEDULER_PID_FILE="${RUN_DIR}/eidos_scheduler.pid"
LOCAL_AGENT_PID_FILE="${RUN_DIR}/eidos_local_agent.pid"
OLLAMA_QWEN_PID_FILE="${RUN_DIR}/ollama_qwen.pid"
OLLAMA_EMBED_PID_FILE="${RUN_DIR}/ollama_embedding.pid"
MCP_LOG_FILE="${FORGE_ROOT}/doc_forge/mcp_server.log"
DOC_LOG_FILE="${FORGE_ROOT}/doc_forge/orchestrator.log"
ATLAS_LOG_FILE="${FORGE_ROOT}/web_interface_forge/eidos_atlas.log"
SCHEDULER_LOG_FILE="${FORGE_ROOT}/logs/eidos_scheduler.log"
LOCAL_AGENT_LOG_FILE="${FORGE_ROOT}/logs/eidos_local_agent.log"
OLLAMA_QWEN_LOG_FILE="${FORGE_ROOT}/logs/ollama_qwen.log"
OLLAMA_EMBED_LOG_FILE="${FORGE_ROOT}/logs/ollama_embedding.log"

_registry_port_default() {
    local service="$1"
    local fallback="$2"
    if [ -x "${PYTHON_BIN}" ] && [ -f "${PORT_REGISTRY_SCRIPT}" ]; then
        "${PYTHON_BIN}" "${PORT_REGISTRY_SCRIPT}" get --service "${service}" --field port --default "${fallback}" 2>/dev/null || printf '%s' "${fallback}"
        return 0
    fi
    printf '%s' "${fallback}"
}

DEFAULT_MCP_PORT="$(_registry_port_default eidos_mcp 8928)"
DEFAULT_DOC_PORT="$(_registry_port_default doc_forge_dashboard 8930)"
DEFAULT_ATLAS_PORT="$(_registry_port_default eidos_atlas_dashboard 8936)"
DEFAULT_OLLAMA_QWEN_PORT="$(_registry_port_default ollama_qwen_http 8938)"
DEFAULT_OLLAMA_EMBED_PORT="$(_registry_port_default ollama_embedding_http 8940)"
RUNIT_SERVICE_ROOT="${EIDOS_RUNIT_SERVICE_DIR:-${PREFIX:-/data/data/com.termux/files/usr}/var/service}"

MCP_PORT="${EIDOS_MCP_PORT:-${DEFAULT_MCP_PORT}}"
MCP_HEALTH_URL="${EIDOS_MCP_HEALTH_URL:-http://127.0.0.1:${MCP_PORT}/health}"
DOC_PORT="${EIDOS_DOC_FORGE_PORT:-${DEFAULT_DOC_PORT}}"
DOC_HEALTH_URL="${EIDOS_DOC_FORGE_HEALTH_URL:-http://127.0.0.1:${DOC_PORT}/health}"
ATLAS_PORT="${EIDOS_ATLAS_PORT:-${DEFAULT_ATLAS_PORT}}"
ATLAS_HEALTH_URL="${EIDOS_ATLAS_HEALTH_URL:-http://127.0.0.1:${ATLAS_PORT}/health}"
OLLAMA_QWEN_PORT="${EIDOS_OLLAMA_QWEN_PORT:-${DEFAULT_OLLAMA_QWEN_PORT}}"
OLLAMA_QWEN_HEALTH_URL="${EIDOS_OLLAMA_QWEN_HEALTH_URL:-http://127.0.0.1:${OLLAMA_QWEN_PORT}/api/tags}"
OLLAMA_EMBED_PORT="${EIDOS_OLLAMA_EMBEDDING_PORT:-${DEFAULT_OLLAMA_EMBED_PORT}}"
OLLAMA_EMBED_HEALTH_URL="${EIDOS_OLLAMA_EMBED_HEALTH_URL:-http://127.0.0.1:${OLLAMA_EMBED_PORT}/api/tags}"

ENABLE_DOC_FORGE_AUTOSTART="${EIDOS_ENABLE_DOC_FORGE_AUTOSTART:-1}"
ENABLE_ATLAS_AUTOSTART="${EIDOS_ENABLE_ATLAS_AUTOSTART:-1}"
ENABLE_SCHEDULER_AUTOSTART="${EIDOS_ENABLE_SCHEDULER_AUTOSTART:-1}"
ENABLE_LOCAL_AGENT_AUTOSTART="${EIDOS_ENABLE_LOCAL_AGENT_AUTOSTART:-1}"
ENABLE_OLLAMA_AUTOSTART="${EIDOS_ENABLE_OLLAMA_AUTOSTART:-1}"

mkdir -p "${RUN_DIR}"

_log() {
    printf '[eidos-services] %s\n' "$*"
}

_pid_alive() {
    local pid="${1:-}"
    [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1
}

_pid_matches() {
    local pid="${1:-}"
    local expected="${2:-}"
    [ -n "${pid}" ] || return 1
    [ -n "${expected}" ] || return 0
    local cmdline=""
    cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [ -n "${cmdline}" ] && printf '%s' "${cmdline}" | grep -Fq "${expected}"
}

_read_count() {
    local count=0
    if [ -f "${SHELL_COUNT_FILE}" ]; then
        count="$(cat "${SHELL_COUNT_FILE}" 2>/dev/null || echo 0)"
    fi
    case "${count}" in
        ''|*[!0-9]*) count=0 ;;
    esac
    printf '%s' "${count}"
}

_write_count() {
    printf '%s\n' "${1:-0}" > "${SHELL_COUNT_FILE}"
}

_port_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
            return 0
        fi
    fi
    if [ -x "${PYTHON_BIN}" ]; then
        "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.4)
try:
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", int("${port}"))) == 0 else 1)
finally:
    s.close()
PY
        return $?
    fi
    return 1
}

_wait_http_ok() {
    local url="$1"
    local timeout_s="${2:-12}"
    local waited=0
    if ! command -v curl >/dev/null 2>&1; then
        return 1
    fi
    while [ "${waited}" -lt "${timeout_s}" ]; do
        if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
            return 0
        fi
        waited=$((waited + 1))
        sleep 1
    done
    return 1
}

_http_ok() {
    local url="$1"
    if ! command -v curl >/dev/null 2>&1; then
        return 1
    fi
    curl -fsS --max-time 2 "${url}" >/dev/null 2>&1
}

_is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

_normalize_service_target() {
    case "${1:-all}" in
        ""|all) printf 'all' ;;
        ollama-qwen|qwen|eidos-ollama-qwen) printf 'ollama-qwen' ;;
        ollama-embedding|embedding|eidos-ollama-embedding) printf 'ollama-embedding' ;;
        mcp|eidos-mcp) printf 'mcp' ;;
        doc-forge|doc_forge|eidos-doc-forge) printf 'doc-forge' ;;
        atlas|dashboard|eidos-atlas) printf 'atlas' ;;
        scheduler|eidos-scheduler) printf 'scheduler' ;;
        local-agent|local_agent|agent|eidos-local-agent) printf 'local-agent' ;;
        *) return 1 ;;
    esac
}

_target_selected() {
    local target="$1"
    local key="$2"
    [ "${target}" = "all" ] || [ "${target}" = "${key}" ]
}

_runit_service_path() {
    printf '%s/%s' "${RUNIT_SERVICE_ROOT}" "$1"
}

_runit_service_installed() {
    local service_key="$1"
    command -v sv >/dev/null 2>&1 || return 1
    [ -d "$(_runit_service_path "${service_key}")" ]
}

_runit_start() {
    local service_key="$1"
    local service_dir
    service_dir="$(_runit_service_path "${service_key}")"
    _runit_service_installed "${service_key}" || return 1
    rm -f "${service_dir}/down"
    sv cont "${service_dir}" >/dev/null 2>&1 || true
    sv up "${service_dir}" >/dev/null 2>&1 || true
    sv cont "${service_dir}" >/dev/null 2>&1 || true
    sleep 1
    sv status "${service_dir}" >/dev/null 2>&1
}

_runit_stop() {
    local service_key="$1"
    local service_dir
    service_dir="$(_runit_service_path "${service_key}")"
    _runit_service_installed "${service_key}" || return 1
    sv cont "${service_dir}" >/dev/null 2>&1 || true
    sv down "${service_dir}" >/dev/null 2>&1 || true
    : > "${service_dir}/down"
    return 0
}

_runit_pause() {
    local service_key="$1"
    local service_dir
    service_dir="$(_runit_service_path "${service_key}")"
    _runit_service_installed "${service_key}" || return 1
    sv pause "${service_dir}" >/dev/null 2>&1 || true
    sleep 1
    return 0
}

_runit_resume() {
    local service_key="$1"
    local service_dir
    service_dir="$(_runit_service_path "${service_key}")"
    _runit_service_installed "${service_key}" || return 1
    sv cont "${service_dir}" >/dev/null 2>&1 || true
    sleep 1
    return 0
}

_runit_status() {
    local service_key="$1"
    local service_dir
    service_dir="$(_runit_service_path "${service_key}")"
    _runit_service_installed "${service_key}" || return 1
    sv status "${service_dir}" 2>&1 || true
}

_pid_state() {
    local pid="${1:-}"
    [ -n "${pid}" ] || return 1
    ps -p "${pid}" -o stat= 2>/dev/null | tr -d '[:space:]'
}

_pid_paused() {
    local pid="${1:-}"
    local state=""
    state="$(_pid_state "${pid}" || true)"
    [ -n "${state}" ] && printf '%s' "${state}" | grep -Eq '^T'
}

_ensure_termux_service_supervisor() {
    command -v sv >/dev/null 2>&1 || return 0
    [ -f "${START_SERVICES_SH}" ] || return 0
    # shellcheck source=/dev/null
    . "${START_SERVICES_SH}" >/dev/null 2>&1 || true
}

_start_service() {
    local service_name="$1"
    local script_path="$2"
    local pid_file="$3"
    local log_file="$4"
    local port_check="${5:-}"
    local expected_cmd="${6:-}"
    local runit_service="${7:-}"

    _ensure_termux_service_supervisor
    if [ -n "${runit_service}" ] && _runit_start "${runit_service}"; then
        _log "${service_name}: started via runit (${runit_service})."
        return 0
    fi

    mkdir -p "$(dirname "${pid_file}")" "$(dirname "${log_file}")"

    if [ -f "${pid_file}" ]; then
        local existing_pid
        existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if _pid_alive "${existing_pid}" && _pid_matches "${existing_pid}" "${expected_cmd}"; then
            return 0
        fi
        rm -f "${pid_file}"
    fi

    if [ -n "${port_check}" ] && _port_listening "${port_check}"; then
        _log "${service_name}: already listening on ${port_check} (external process)."
        return 0
    fi

    (
        cd "${FORGE_ROOT}" || exit 1
        EIDOS_DISABLE_NOTIFICATIONS=1 nohup bash "${script_path}" >> "${log_file}" 2>&1 &
        echo "$!" > "${pid_file}"
    )

    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    sleep 1
    if ! _pid_alive "${pid}"; then
        rm -f "${pid_file}"
        _log "${service_name}: failed to start."
        return 1
    fi
    _log "${service_name}: started (pid ${pid})."
}

_stop_service() {
    local service_name="$1"
    local pid_file="$2"
    local expected_cmd="${3:-}"
    local runit_service="${4:-}"

    if [ -n "${runit_service}" ] && _runit_stop "${runit_service}"; then
        _log "${service_name}: stopped via runit (${runit_service})."
    fi

    [ -f "${pid_file}" ] || return 0

    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if _pid_alive "${pid}" && _pid_matches "${pid}" "${expected_cmd}"; then
        kill "${pid}" >/dev/null 2>&1 || true
        sleep 1
        _pid_alive "${pid}" && kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${pid_file}"
    _log "${service_name}: stopped."
}

_pause_service() {
    local service_name="$1"
    local pid_file="$2"
    local expected_cmd="${3:-}"
    local runit_service="${4:-}"

    if [ -n "${runit_service}" ] && _runit_pause "${runit_service}"; then
        _log "${service_name}: paused via runit (${runit_service})."
        return 0
    fi

    [ -f "${pid_file}" ] || return 0
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if _pid_alive "${pid}" && _pid_matches "${pid}" "${expected_cmd}"; then
        kill -STOP "${pid}" >/dev/null 2>&1 || true
        _log "${service_name}: paused."
    fi
}

_resume_service() {
    local service_name="$1"
    local pid_file="$2"
    local expected_cmd="${3:-}"
    local runit_service="${4:-}"

    if [ -n "${runit_service}" ] && _runit_resume "${runit_service}"; then
        _log "${service_name}: resumed via runit (${runit_service})."
        return 0
    fi

    [ -f "${pid_file}" ] || return 0
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if _pid_alive "${pid}" && _pid_matches "${pid}" "${expected_cmd}"; then
        kill -CONT "${pid}" >/dev/null 2>&1 || true
        _log "${service_name}: resumed."
    fi
}

_status_service() {
    local service_name="$1"
    local pid_file="$2"
    local port="${3:-}"
    local expected_cmd="${4:-}"
    local health_url="${5:-}"
    local runit_service="${6:-}"
    local status="stopped"
    local pid=""
    local runit_status=""

    if [ -n "${runit_service}" ] && _runit_service_installed "${runit_service}"; then
        runit_status="$(_runit_status "${runit_service}")"
        if printf '%s' "${runit_status}" | grep -Fq 'warning:'; then
            runit_status=""
        fi
        if [ -n "${runit_status}" ] && printf '%s' "${runit_status}" | grep -Fq 'paused'; then
            status="paused(runit ${runit_status})"
        elif [ -n "${runit_status}" ] && printf '%s' "${runit_status}" | grep -Eq '^pause:'; then
            status="paused(runit ${runit_status})"
        elif [ -n "${runit_status}" ] && printf '%s' "${runit_status}" | grep -Eq '^run:'; then
            status="runit ${runit_status}"
        fi
    fi

    if [ -f "${pid_file}" ]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if _pid_alive "${pid}" && _pid_matches "${pid}" "${expected_cmd}"; then
            if _pid_paused "${pid}"; then
                status="paused(managed pid=${pid})"
            else
                status="running(managed pid=${pid})"
            fi
        else
            rm -f "${pid_file}"
        fi
    fi
    if [ -n "${port}" ] && _port_listening "${port}" && [ "${status}" = "stopped" ]; then
        status="running(external port=${port})"
    fi
    if [ -n "${health_url}" ] && _http_ok "${health_url}" && [ "${status}" = "stopped" ]; then
        status="running(external health=${health_url})"
    fi
    if [ "${status}" = "stopped" ] && [ -n "${runit_status}" ]; then
        status="runit ${runit_status}"
    fi
    printf '%s: %s\n' "${service_name}" "${status}"
}

_with_lock() {
    local action="$1"
    shift
    (
        flock -x 200
        "${action}" "$@"
    ) 200>"${LOCK_FILE}"
}

_increment_shell_count_locked() {
    local count
    count="$(_read_count)"
    count=$((count + 1))
    _write_count "${count}"
    printf '%s\n' "${count}"
}

_decrement_shell_count_locked() {
    local count
    count="$(_read_count)"
    if [ "${count}" -gt 0 ]; then
        count=$((count - 1))
    fi
    _write_count "${count}"
    printf '%s\n' "${count}"
}

cmd="${1:-status}"
service_target="$(_normalize_service_target "${2:-all}" 2>/dev/null || true)"
if [ -z "${service_target}" ]; then
    echo "Invalid service target: ${2:-}" >&2
    exit 2
fi
case "${cmd}" in
    start-shell)
        _with_lock _increment_shell_count_locked >/dev/null
        if _is_truthy "${ENABLE_OLLAMA_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/run_ollama_qwen.sh" ] && [ -x "${FORGE_ROOT}/scripts/run_ollama_embedding.sh" ]; then
            _start_service "Eidos Ollama Qwen" "${FORGE_ROOT}/scripts/run_ollama_qwen.sh" "${OLLAMA_QWEN_PID_FILE}" "${OLLAMA_QWEN_LOG_FILE}" "${OLLAMA_QWEN_PORT}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen" || true
            _start_service "Eidos Ollama Embedding" "${FORGE_ROOT}/scripts/run_ollama_embedding.sh" "${OLLAMA_EMBED_PID_FILE}" "${OLLAMA_EMBED_LOG_FILE}" "${OLLAMA_EMBED_PORT}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding" || true
            _wait_http_ok "${OLLAMA_QWEN_HEALTH_URL}" 30 || _log "warning: Ollama qwen health check failed at ${OLLAMA_QWEN_HEALTH_URL}."
            _wait_http_ok "${OLLAMA_EMBED_HEALTH_URL}" 30 || _log "warning: Ollama embedding health check failed at ${OLLAMA_EMBED_HEALTH_URL}."
        fi
        _start_service "Eidos MCP Server" "${FORGE_ROOT}/eidos_mcp/run_server.sh" "${MCP_PID_FILE}" "${MCP_LOG_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh" "eidos-mcp" || true
        if _is_truthy "${ENABLE_DOC_FORGE_AUTOSTART}" && [ -x "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" ]; then
            _start_service "Eidos Documentation Forge" "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" "${DOC_PID_FILE}" "${DOC_LOG_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge" || true
            if ! _wait_http_ok "${DOC_HEALTH_URL}" 20; then
                _log "warning: Doc Forge health check failed at ${DOC_HEALTH_URL}."
            fi
        fi
        if _is_truthy "${ENABLE_ATLAS_AUTOSTART}" && [ -x "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" ]; then
            _start_service "Eidos Atlas Dashboard" "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_PID_FILE}" "${ATLAS_LOG_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas" || true
        fi
        if _is_truthy "${ENABLE_SCHEDULER_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/eidos_scheduler.py" ]; then
            _start_service "Eidos Scheduler" "${FORGE_ROOT}/scripts/run_eidos_scheduler.sh" "${SCHEDULER_PID_FILE}" "${SCHEDULER_LOG_FILE}" "" "scripts/run_eidos_scheduler.sh" "eidos-scheduler" || true
        fi
        if _is_truthy "${ENABLE_LOCAL_AGENT_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/run_local_mcp_agent.sh" ]; then
            _start_service "Eidos Local Agent" "${FORGE_ROOT}/scripts/run_local_mcp_agent.sh" "${LOCAL_AGENT_PID_FILE}" "${LOCAL_AGENT_LOG_FILE}" "" "scripts/run_local_mcp_agent.sh" "eidos-local-agent" || true
        fi

        if ! _wait_http_ok "${MCP_HEALTH_URL}" 15; then
            _log "warning: MCP health check failed at ${MCP_HEALTH_URL}."
        fi
        ;;
    exit-shell)
        remaining_count="$(_with_lock _decrement_shell_count_locked)"
        if [ "${remaining_count}" -eq 0 ]; then
            _stop_service "Eidos Ollama Qwen" "${OLLAMA_QWEN_PID_FILE}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen"
            _stop_service "Eidos Ollama Embedding" "${OLLAMA_EMBED_PID_FILE}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding"
            _stop_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh" "eidos-mcp"
            _stop_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge"
            _stop_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas"
            _stop_service "Eidos Scheduler" "${SCHEDULER_PID_FILE}" "scripts/run_eidos_scheduler.sh" "eidos-scheduler"
            _stop_service "Eidos Local Agent" "${LOCAL_AGENT_PID_FILE}" "scripts/run_local_mcp_agent.sh" "eidos-local-agent"
        fi
        ;;
    start)
        if _target_selected "${service_target}" "ollama-qwen" && _is_truthy "${ENABLE_OLLAMA_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/run_ollama_qwen.sh" ]; then
            _start_service "Eidos Ollama Qwen" "${FORGE_ROOT}/scripts/run_ollama_qwen.sh" "${OLLAMA_QWEN_PID_FILE}" "${OLLAMA_QWEN_LOG_FILE}" "${OLLAMA_QWEN_PORT}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen"
            _wait_http_ok "${OLLAMA_QWEN_HEALTH_URL}" 30 || {
                _log "warning: Ollama qwen health check failed at ${OLLAMA_QWEN_HEALTH_URL}."
            }
        fi
        if _target_selected "${service_target}" "ollama-embedding" && _is_truthy "${ENABLE_OLLAMA_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/run_ollama_embedding.sh" ]; then
            _start_service "Eidos Ollama Embedding" "${FORGE_ROOT}/scripts/run_ollama_embedding.sh" "${OLLAMA_EMBED_PID_FILE}" "${OLLAMA_EMBED_LOG_FILE}" "${OLLAMA_EMBED_PORT}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding"
            _wait_http_ok "${OLLAMA_EMBED_HEALTH_URL}" 30 || {
                _log "warning: Ollama embedding health check failed at ${OLLAMA_EMBED_HEALTH_URL}."
            }
        fi
        if _target_selected "${service_target}" "mcp"; then
            _start_service "Eidos MCP Server" "${FORGE_ROOT}/eidos_mcp/run_server.sh" "${MCP_PID_FILE}" "${MCP_LOG_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh" "eidos-mcp"
        fi
        if _target_selected "${service_target}" "doc-forge" && _is_truthy "${ENABLE_DOC_FORGE_AUTOSTART}" && [ -x "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" ]; then
            _start_service "Eidos Documentation Forge" "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" "${DOC_PID_FILE}" "${DOC_LOG_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge"
            _wait_http_ok "${DOC_HEALTH_URL}" 20 || {
                _log "warning: Doc Forge health check failed at ${DOC_HEALTH_URL}."
            }
        fi
        if _target_selected "${service_target}" "atlas" && _is_truthy "${ENABLE_ATLAS_AUTOSTART}" && [ -x "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" ]; then
            _start_service "Eidos Atlas Dashboard" "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_PID_FILE}" "${ATLAS_LOG_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas"
            _wait_http_ok "${ATLAS_HEALTH_URL}" 15 || {
                _log "warning: Atlas Dashboard health check failed at ${ATLAS_HEALTH_URL}."
            }
        fi
        if _target_selected "${service_target}" "scheduler" && _is_truthy "${ENABLE_SCHEDULER_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/eidos_scheduler.py" ]; then
            _start_service "Eidos Scheduler" "${FORGE_ROOT}/scripts/run_eidos_scheduler.sh" "${SCHEDULER_PID_FILE}" "${SCHEDULER_LOG_FILE}" "" "scripts/run_eidos_scheduler.sh" "eidos-scheduler"
        fi
        if _target_selected "${service_target}" "local-agent" && _is_truthy "${ENABLE_LOCAL_AGENT_AUTOSTART}" && [ -x "${FORGE_ROOT}/scripts/run_local_mcp_agent.sh" ]; then
            _start_service "Eidos Local Agent" "${FORGE_ROOT}/scripts/run_local_mcp_agent.sh" "${LOCAL_AGENT_PID_FILE}" "${LOCAL_AGENT_LOG_FILE}" "" "scripts/run_local_mcp_agent.sh" "eidos-local-agent"
        fi

        if _target_selected "${service_target}" "mcp"; then
            _wait_http_ok "${MCP_HEALTH_URL}" 15 || {
            _log "warning: MCP health check failed at ${MCP_HEALTH_URL}."
            exit 1
            }
        fi
        ;;
    stop)
        if _target_selected "${service_target}" "ollama-qwen"; then _stop_service "Eidos Ollama Qwen" "${OLLAMA_QWEN_PID_FILE}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen"; fi
        if _target_selected "${service_target}" "ollama-embedding"; then _stop_service "Eidos Ollama Embedding" "${OLLAMA_EMBED_PID_FILE}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding"; fi
        if _target_selected "${service_target}" "mcp"; then _stop_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh" "eidos-mcp"; fi
        if _target_selected "${service_target}" "doc-forge"; then _stop_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge"; fi
        if _target_selected "${service_target}" "atlas"; then _stop_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas"; fi
        if _target_selected "${service_target}" "scheduler"; then _stop_service "Eidos Scheduler" "${SCHEDULER_PID_FILE}" "scripts/run_eidos_scheduler.sh" "eidos-scheduler"; fi
        if _target_selected "${service_target}" "local-agent"; then _stop_service "Eidos Local Agent" "${LOCAL_AGENT_PID_FILE}" "scripts/run_local_mcp_agent.sh" "eidos-local-agent"; fi
        ;;
    pause)
        if _target_selected "${service_target}" "ollama-qwen"; then _pause_service "Eidos Ollama Qwen" "${OLLAMA_QWEN_PID_FILE}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen"; fi
        if _target_selected "${service_target}" "ollama-embedding"; then _pause_service "Eidos Ollama Embedding" "${OLLAMA_EMBED_PID_FILE}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding"; fi
        if _target_selected "${service_target}" "mcp"; then _pause_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh" "eidos-mcp"; fi
        if _target_selected "${service_target}" "doc-forge"; then _pause_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge"; fi
        if _target_selected "${service_target}" "atlas"; then _pause_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas"; fi
        if _target_selected "${service_target}" "scheduler"; then _pause_service "Eidos Scheduler" "${SCHEDULER_PID_FILE}" "scripts/run_eidos_scheduler.sh" "eidos-scheduler"; fi
        if _target_selected "${service_target}" "local-agent"; then _pause_service "Eidos Local Agent" "${LOCAL_AGENT_PID_FILE}" "scripts/run_local_mcp_agent.sh" "eidos-local-agent"; fi
        ;;
    resume)
        if _target_selected "${service_target}" "ollama-qwen"; then _resume_service "Eidos Ollama Qwen" "${OLLAMA_QWEN_PID_FILE}" "scripts/run_ollama_qwen.sh" "eidos-ollama-qwen"; fi
        if _target_selected "${service_target}" "ollama-embedding"; then _resume_service "Eidos Ollama Embedding" "${OLLAMA_EMBED_PID_FILE}" "scripts/run_ollama_embedding.sh" "eidos-ollama-embedding"; fi
        if _target_selected "${service_target}" "mcp"; then _resume_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh" "eidos-mcp"; fi
        if _target_selected "${service_target}" "doc-forge"; then _resume_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh" "eidos-doc-forge"; fi
        if _target_selected "${service_target}" "atlas"; then _resume_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh" "eidos-atlas"; fi
        if _target_selected "${service_target}" "scheduler"; then _resume_service "Eidos Scheduler" "${SCHEDULER_PID_FILE}" "scripts/run_eidos_scheduler.sh" "eidos-scheduler"; fi
        if _target_selected "${service_target}" "local-agent"; then _resume_service "Eidos Local Agent" "${LOCAL_AGENT_PID_FILE}" "scripts/run_local_mcp_agent.sh" "eidos-local-agent"; fi
        ;;
    restart)
        "$0" stop "${service_target}"
        "$0" start "${service_target}"
        ;;
    low-load)
        "$0" start mcp || true
        "$0" start atlas || true
        "$0" pause scheduler || true
        "$0" pause local-agent || true
        "$0" stop ollama-qwen || true
        "$0" stop ollama-embedding || true
        "$0" stop doc-forge || true
        ;;
    restore-standard)
        "$0" start mcp || true
        "$0" start atlas || true
        "$0" start doc-forge || true
        "$0" start ollama-qwen || true
        "$0" start ollama-embedding || true
        "$0" resume scheduler || true
        "$0" resume local-agent || true
        ;;
    install-runit)
        exec "${FORGE_ROOT}/scripts/install_termux_runit_services.sh"
        ;;
    status)
        if _target_selected "${service_target}" "ollama-qwen"; then _status_service "Eidos Ollama Qwen" "${OLLAMA_QWEN_PID_FILE}" "${OLLAMA_QWEN_PORT}" "scripts/run_ollama_qwen.sh" "${OLLAMA_QWEN_HEALTH_URL}" "eidos-ollama-qwen"; fi
        if _target_selected "${service_target}" "ollama-embedding"; then _status_service "Eidos Ollama Embedding" "${OLLAMA_EMBED_PID_FILE}" "${OLLAMA_EMBED_PORT}" "scripts/run_ollama_embedding.sh" "${OLLAMA_EMBED_HEALTH_URL}" "eidos-ollama-embedding"; fi
        if _target_selected "${service_target}" "mcp"; then _status_service "Eidos MCP Server" "${MCP_PID_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh" "${MCP_HEALTH_URL}" "eidos-mcp"; fi
        if _target_selected "${service_target}" "doc-forge"; then _status_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh" "${DOC_HEALTH_URL}" "eidos-doc-forge"; fi
        if _target_selected "${service_target}" "atlas"; then _status_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_HEALTH_URL}" "eidos-atlas"; fi
        if _target_selected "${service_target}" "scheduler"; then _status_service "Eidos Scheduler" "${SCHEDULER_PID_FILE}" "" "scripts/run_eidos_scheduler.sh" "" "eidos-scheduler"; fi
        if _target_selected "${service_target}" "local-agent"; then _status_service "Eidos Local Agent" "${LOCAL_AGENT_PID_FILE}" "" "scripts/run_local_mcp_agent.sh" "" "eidos-local-agent"; fi
        printf 'Interactive shell refcount: %s\n' "$(_read_count)"
        ;;
    *)
        echo "Usage: $0 {start-shell|exit-shell|start|stop|pause|resume|restart|low-load|restore-standard|install-runit|status} [all|ollama-qwen|ollama-embedding|mcp|doc-forge|atlas|scheduler|local-agent]" >&2
        exit 2
        ;;
esac
