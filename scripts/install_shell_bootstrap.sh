#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${FORGE_ROOT}/shell/templates/bashrc.sh"
TARGET_HOME="${HOME}"
TARGET_BASHRC="${TARGET_HOME}/.bashrc"
BACKUP_DIR="${FORGE_ROOT}/backups/shell_bootstrap"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${BACKUP_DIR}"
if [ -f "${TARGET_BASHRC}" ]; then
    cp "${TARGET_BASHRC}" "${BACKUP_DIR}/bashrc.${STAMP}.bak"
fi
install -m 0644 "${TEMPLATE}" "${TARGET_BASHRC}"
printf '%s\n' "${TARGET_BASHRC}"
