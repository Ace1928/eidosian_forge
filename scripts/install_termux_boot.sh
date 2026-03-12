#!/usr/bin/env bash
set -euo pipefail

TARGET_HOME="${HOME}"
BOOT_DIR="${TARGET_HOME}/.termux/boot"
FORGE_ROOT="${EIDOS_FORGE_ROOT:-${TARGET_HOME}/eidosian_forge}"
BOOT_SCRIPT="${FORGE_ROOT}/scripts/eidos_termux_boot.sh"
WRAPPER="${BOOT_DIR}/00-eidos-boot"
RUNIT_INSTALLER="${FORGE_ROOT}/scripts/install_termux_runit_services.sh"
AUTO_INSTALL_RUNIT="${EIDOS_INSTALL_RUNIT_ON_BOOT_INSTALL:-1}"

mkdir -p "${BOOT_DIR}"
if [ "${AUTO_INSTALL_RUNIT}" = "1" ] && [ -x "${RUNIT_INSTALLER}" ]; then
    "${RUNIT_INSTALLER}" >/dev/null 2>&1 || true
fi
cat > "${WRAPPER}" <<BOOT
#!/data/data/com.termux/files/usr/bin/bash
exec "${BOOT_SCRIPT}" "\$@"
BOOT
chmod 0700 "${WRAPPER}"
printf '%s\n' "${WRAPPER}"
