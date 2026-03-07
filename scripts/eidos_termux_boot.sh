#!/usr/bin/env bash
set -euo pipefail

FORGE_ROOT="${EIDOS_FORGE_ROOT:-$HOME/eidosian_forge}"
SERVICES_SCRIPT="${FORGE_ROOT}/scripts/eidos_termux_services.sh"
START_SERVICES_SH="${PREFIX:-/data/data/com.termux/files/usr}/etc/profile.d/start-services.sh"
LOCK_DIR="${HOME}/.eidosian/run"
LOCK_FILE="${LOCK_DIR}/termux_boot.lock"

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    exit 0
fi

if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock >/dev/null 2>&1 || true
fi

if [ -f "${START_SERVICES_SH}" ]; then
    # shellcheck source=/dev/null
    . "${START_SERVICES_SH}"
fi

if [ -x "${SERVICES_SCRIPT}" ]; then
    EIDOS_DISABLE_NOTIFICATIONS=1 "${SERVICES_SCRIPT}" start >/dev/null 2>&1 || true
fi
