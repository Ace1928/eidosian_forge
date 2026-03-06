#!/usr/bin/env bash
set -euo pipefail

NOW_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
HOME_DIR="${HOME:-/data/data/com.termux/files/home}"
FORGE_ROOT="${EIDOS_FORGE_ROOT:-${HOME_DIR}/eidosian_forge}"
BACKUP_ROOT="${FORGE_ROOT}/backups/termux_startup/${NOW_UTC}"

mkdir -p "${BACKUP_ROOT}/home" "${BACKUP_ROOT}/termux"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dest}")"
    cp -a "${src}" "${dest}"
  fi
}

copy_if_exists "${HOME_DIR}/.bashrc" "${BACKUP_ROOT}/home/.bashrc"
copy_if_exists "${HOME_DIR}/.termux" "${BACKUP_ROOT}/termux/.termux"
copy_if_exists "${HOME_DIR}/scripts" "${BACKUP_ROOT}/home/scripts"
copy_if_exists "${FORGE_ROOT}/eidos_env.sh" "${BACKUP_ROOT}/forge/eidos_env.sh"
copy_if_exists "${FORGE_ROOT}/shell" "${BACKUP_ROOT}/forge/shell"

cat > "${BACKUP_ROOT}/manifest.json" <<EOF
{
  "created_at_utc": "${NOW_UTC}",
  "home_dir": "${HOME_DIR}",
  "forge_root": "${FORGE_ROOT}",
  "paths": [
    "${HOME_DIR}/.bashrc",
    "${HOME_DIR}/.termux",
    "${HOME_DIR}/scripts",
    "${FORGE_ROOT}/eidos_env.sh",
    "${FORGE_ROOT}/shell"
  ]
}
EOF

printf '%s\n' "${BACKUP_ROOT}"
