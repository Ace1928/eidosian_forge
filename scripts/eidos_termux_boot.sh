#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-$HOME/eidosian_forge}"
SERVICES_SCRIPT="${FORGE_ROOT}/scripts/eidos_termux_services.sh"
RUNIT_INSTALLER="${FORGE_ROOT}/scripts/install_termux_runit_services.sh"
CAPABILITIES_WRITER="${FORGE_ROOT}/scripts/write_runtime_capabilities.py"
START_SERVICES_SH="${PREFIX:-/data/data/com.termux/files/usr}/etc/profile.d/start-services.sh"
LOCK_DIR="${HOME}/.eidosian/run"
LOCK_FILE="${LOCK_DIR}/termux_boot.lock"
BOOT_STATE_PATH="${FORGE_ROOT}/data/runtime/termux_boot_status.json"
RUNIT_SERVICE_ROOT="${EIDOS_RUNIT_SERVICE_DIR:-${PREFIX:-/data/data/com.termux/files/usr}/var/service}"

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    exit 0
fi

if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock >/dev/null 2>&1 || true
fi

if [ -x "${RUNIT_INSTALLER}" ] && [ ! -d "${RUNIT_SERVICE_ROOT}/eidos-scheduler" ]; then
    "${RUNIT_INSTALLER}" >/dev/null 2>&1 || true
fi

if [ -f "${START_SERVICES_SH}" ]; then
    # shellcheck source=/dev/null
    . "${START_SERVICES_SH}"
fi

if [ -x "${SERVICES_SCRIPT}" ]; then
    EIDOS_DISABLE_NOTIFICATIONS=1 "${SERVICES_SCRIPT}" start >/dev/null 2>&1 || true
fi

if [ -x "${FORGE_ROOT}/eidosian_venv/bin/python" ] && [ -f "${CAPABILITIES_WRITER}" ]; then
    PYTHONPATH="${FORGE_ROOT}/lib:${PYTHONPATH:-}" "${FORGE_ROOT}/eidosian_venv/bin/python" "${CAPABILITIES_WRITER}" >/dev/null 2>&1 || true
fi

mkdir -p "$(dirname "${BOOT_STATE_PATH}")"
cat > "${BOOT_STATE_PATH}" <<EOF
{
  "contract": "eidos.termux_boot_status.v1",
  "booted_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "forge_root": "${FORGE_ROOT}",
  "runit_service_root": "${RUNIT_SERVICE_ROOT}",
  "wake_lock_requested": true,
  "start_services_sourced": $([ -f "${START_SERVICES_SH}" ] && printf true || printf false),
  "runit_installer_present": $([ -x "${RUNIT_INSTALLER}" ] && printf true || printf false),
  "services_script_present": $([ -x "${SERVICES_SCRIPT}" ] && printf true || printf false)
}
EOF
