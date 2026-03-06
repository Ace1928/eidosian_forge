#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: $0 <backup_dir>" >&2
  exit 2
fi

BACKUP_DIR="$1"
HOME_DIR="${HOME:-/data/data/com.termux/files/home}"
FORGE_ROOT="${EIDOS_FORGE_ROOT:-${HOME_DIR}/eidosian_forge}"

if [ ! -d "${BACKUP_DIR}" ]; then
  echo "[restore] backup dir not found: ${BACKUP_DIR}" >&2
  exit 1
fi

restore_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    rm -rf "${dest}"
    mkdir -p "$(dirname "${dest}")"
    cp -a "${src}" "${dest}"
  fi
}

restore_if_exists "${BACKUP_DIR}/home/.bashrc" "${HOME_DIR}/.bashrc"
restore_if_exists "${BACKUP_DIR}/termux/.termux" "${HOME_DIR}/.termux"
restore_if_exists "${BACKUP_DIR}/home/scripts" "${HOME_DIR}/scripts"
restore_if_exists "${BACKUP_DIR}/forge/eidos_env.sh" "${FORGE_ROOT}/eidos_env.sh"
restore_if_exists "${BACKUP_DIR}/forge/shell" "${FORGE_ROOT}/shell"

echo "[restore] restored startup state from ${BACKUP_DIR}"
