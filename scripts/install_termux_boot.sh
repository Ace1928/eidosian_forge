#!/usr/bin/env bash
set -euo pipefail

TARGET_HOME="${HOME}"
BOOT_DIR="${TARGET_HOME}/.termux/boot"
FORGE_ROOT="${EIDOS_FORGE_ROOT:-${TARGET_HOME}/eidosian_forge}"
BOOT_SCRIPT="${FORGE_ROOT}/scripts/eidos_termux_boot.sh"
WRAPPER="${BOOT_DIR}/00-eidos-boot"

mkdir -p "${BOOT_DIR}"
cat > "${WRAPPER}" <<BOOT
#!/data/data/com.termux/files/usr/bin/bash
exec "${BOOT_SCRIPT}" "\$@"
BOOT
chmod 0700 "${WRAPPER}"
printf '%s\n' "${WRAPPER}"
