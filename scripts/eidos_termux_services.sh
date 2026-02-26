#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-/data/data/com.termux/files/home/eidosian_forge}"
PYTHON_BIN="${EIDOS_VENV_PYTHON:-${FORGE_ROOT}/eidosian_venv/bin/python}"
PORT_REGISTRY_SCRIPT="${FORGE_ROOT}/scripts/port_registry.py"
RUN_DIR="${HOME}/.eidosian/run"
LOCK_FILE="${RUN_DIR}/services.lock"
SHELL_COUNT_FILE="${RUN_DIR}/.interactive_shell_count"
MCP_PID_FILE="${RUN_DIR}/eidos_mcp.pid"
DOC_PID_FILE="${RUN_DIR}/doc_forge.pid"
ATLAS_PID_FILE="${RUN_DIR}/eidos_atlas.pid"
MCP_LOG_FILE="${FORGE_ROOT}/doc_forge/mcp_server.log"
DOC_LOG_FILE="${FORGE_ROOT}/doc_forge/orchestrator.log"
ATLAS_LOG_FILE="${FORGE_ROOT}/web_interface_forge/eidos_atlas.log"

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

MCP_PORT="${EIDOS_MCP_PORT:-${DEFAULT_MCP_PORT}}"
MCP_HEALTH_URL="${EIDOS_MCP_HEALTH_URL:-http://127.0.0.1:${MCP_PORT}/health}"
DOC_PORT="${EIDOS_DOC_FORGE_PORT:-${DEFAULT_DOC_PORT}}"
DOC_HEALTH_URL="${EIDOS_DOC_FORGE_HEALTH_URL:-http://127.0.0.1:${DOC_PORT}/health}"
ATLAS_PORT="${EIDOS_ATLAS_PORT:-${DEFAULT_ATLAS_PORT}}"
ATLAS_HEALTH_URL="${EIDOS_ATLAS_HEALTH_URL:-http://127.0.0.1:${ATLAS_PORT}/health}"

ENABLE_DOC_FORGE_AUTOSTART="${EIDOS_ENABLE_DOC_FORGE_AUTOSTART:-1}"
ENABLE_ATLAS_AUTOSTART="${EIDOS_ENABLE_ATLAS_AUTOSTART:-1}"

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

_start_service() {
    local service_name="$1"
    local script_path="$2"
    local pid_file="$3"
    local log_file="$4"
    local port_check="${5:-}"
    local expected_cmd="${6:-}"

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

_status_service() {
    local service_name="$1"
    local pid_file="$2"
    local port="${3:-}"
    local expected_cmd="${4:-}"
    local health_url="${5:-}"
    local status="stopped"
    local pid=""

    if [ -f "${pid_file}" ]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if _pid_alive "${pid}" && _pid_matches "${pid}" "${expected_cmd}"; then
            status="running(managed pid=${pid})"
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
case "${cmd}" in
    start-shell)
        _with_lock _increment_shell_count_locked >/dev/null
        _start_service "Eidos MCP Server" "${FORGE_ROOT}/eidos_mcp/run_server.sh" "${MCP_PID_FILE}" "${MCP_LOG_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh" || true
        if _is_truthy "${ENABLE_DOC_FORGE_AUTOSTART}" && [ -x "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" ]; then
            _start_service "Eidos Documentation Forge" "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" "${DOC_PID_FILE}" "${DOC_LOG_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh" || true
            if ! _wait_http_ok "${DOC_HEALTH_URL}" 20; then
                _log "warning: Doc Forge health check failed at ${DOC_HEALTH_URL}."
            fi
        fi
        if _is_truthy "${ENABLE_ATLAS_AUTOSTART}" && [ -x "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" ]; then
            _start_service "Eidos Atlas Dashboard" "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_PID_FILE}" "${ATLAS_LOG_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh" || true
        fi

        if ! _wait_http_ok "${MCP_HEALTH_URL}" 15; then
            _log "warning: MCP health check failed at ${MCP_HEALTH_URL}."
        fi
        ;;
    exit-shell)
        remaining_count="$(_with_lock _decrement_shell_count_locked)"
        if [ "${remaining_count}" -eq 0 ]; then
            _stop_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh"
            _stop_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh"
            _stop_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh"
        fi
        ;;
    start)
        _start_service "Eidos MCP Server" "${FORGE_ROOT}/eidos_mcp/run_server.sh" "${MCP_PID_FILE}" "${MCP_LOG_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh"
        if _is_truthy "${ENABLE_DOC_FORGE_AUTOSTART}" && [ -x "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" ]; then
            _start_service "Eidos Documentation Forge" "${FORGE_ROOT}/doc_forge/scripts/run_forge.sh" "${DOC_PID_FILE}" "${DOC_LOG_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh"
            _wait_http_ok "${DOC_HEALTH_URL}" 20 || {
                _log "warning: Doc Forge health check failed at ${DOC_HEALTH_URL}."
            }
        fi
        if _is_truthy "${ENABLE_ATLAS_AUTOSTART}" && [ -x "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" ]; then
            _start_service "Eidos Atlas Dashboard" "${FORGE_ROOT}/web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_PID_FILE}" "${ATLAS_LOG_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh"
            _wait_http_ok "${ATLAS_HEALTH_URL}" 15 || {
                _log "warning: Atlas Dashboard health check failed at ${ATLAS_HEALTH_URL}."
            }
        fi

        _wait_http_ok "${MCP_HEALTH_URL}" 15 || {
            _log "warning: MCP health check failed at ${MCP_HEALTH_URL}."
            exit 1
        }
        ;;
    stop)
        _stop_service "Eidos MCP Server" "${MCP_PID_FILE}" "eidos_mcp/run_server.sh"
        _stop_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "doc_forge/scripts/run_forge.sh"
        _stop_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "web_interface_forge/scripts/run_dashboard.sh"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        _status_service "Eidos MCP Server" "${MCP_PID_FILE}" "${MCP_PORT}" "eidos_mcp/run_server.sh" "${MCP_HEALTH_URL}"
        _status_service "Eidos Documentation Forge" "${DOC_PID_FILE}" "${DOC_PORT}" "doc_forge/scripts/run_forge.sh" "${DOC_HEALTH_URL}"
        _status_service "Eidos Atlas Dashboard" "${ATLAS_PID_FILE}" "${ATLAS_PORT}" "web_interface_forge/scripts/run_dashboard.sh" "${ATLAS_HEALTH_URL}"
        printf 'Interactive shell refcount: %s\n' "$(_read_count)"
        ;;
    *)
        echo "Usage: $0 {start-shell|exit-shell|start|stop|restart|status}" >&2
        exit 2
        ;;
esac
