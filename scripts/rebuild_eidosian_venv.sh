#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${FORGE_ROOT}/eidosian_venv"
BACKUP_ROOT="${FORGE_ROOT}/Backups"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
FORCE=0
WITH_REQUIREMENTS=0
SYSTEM_SITE=0

usage() {
  cat <<'EOF'
Usage: scripts/rebuild_eidosian_venv.sh [--force] [--with-requirements] [--system-site-packages]

Options:
  --force              Rebuild even if venv already exists (backs up current venv first)
  --with-requirements  Install dependencies after creating venv
  --system-site-packages
                       Create venv with access to Termux global site-packages
  -h, --help           Show help

Behavior:
  - Idempotent by default: if venv exists and --force is not set, script exits without changes.
  - Rollback-safe: when --force is used, existing venv is moved to Backups/eidosian_venv_<timestamp>.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --with-requirements)
      WITH_REQUIREMENTS=1
      ;;
    --system-site-packages)
      SYSTEM_SITE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No python interpreter found in PATH." >&2
  exit 1
fi

echo "Forge root: ${FORGE_ROOT}"
echo "Python: ${PYTHON_BIN}"

BACKUP_DIR=""
if [[ -d "${VENV_DIR}" ]]; then
  if [[ "${FORCE}" -ne 1 ]]; then
    echo "Venv already exists at ${VENV_DIR}."
    echo "No changes made. Re-run with --force to rebuild."
    exit 0
  fi
  mkdir -p "${BACKUP_ROOT}"
  BACKUP_DIR="${BACKUP_ROOT}/eidosian_venv_$(date +%Y%m%d_%H%M%S)"
  echo "Backing up existing venv -> ${BACKUP_DIR}"
  mv "${VENV_DIR}" "${BACKUP_DIR}"
fi

echo "Creating venv at ${VENV_DIR}"
if [[ "${SYSTEM_SITE}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Upgrading core packaging tools"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ "${WITH_REQUIREMENTS}" -eq 1 ]]; then
  REQ_FILE=""
  if [[ -f "${FORGE_ROOT}/requirements/eidosian_venv_reqs.txt" ]]; then
    REQ_FILE="${FORGE_ROOT}/requirements/eidosian_venv_reqs.txt"
  elif [[ -f "${FORGE_ROOT}/requirements.txt" ]]; then
    REQ_FILE="${FORGE_ROOT}/requirements.txt"
  fi

  if [[ -n "${REQ_FILE}" ]]; then
    echo "Installing dependencies from ${REQ_FILE}"
    "${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"
  else
    echo "No requirements file found. Skipping dependency install."
  fi
fi

echo "Venv rebuild complete."
if [[ -n "${BACKUP_DIR}" ]]; then
  echo "Rollback command:"
  echo "  rm -rf '${VENV_DIR}' && mv '${BACKUP_DIR}' '${VENV_DIR}'"
fi
