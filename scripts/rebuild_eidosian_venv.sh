#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${FORGE_ROOT}/eidosian_venv"
BACKUP_ROOT="${FORGE_ROOT}/Backups"

resolve_host_python() {
  local entry
  local path_python=""

  IFS=':' read -r -a path_entries <<<"${PATH}"
  for entry in "${path_entries[@]}"; do
    [[ -z "${entry}" ]] && continue
    [[ "${entry}" == "${VENV_DIR}/bin" ]] && continue
    if [[ -x "${entry}/python3" ]]; then
      path_python="${entry}/python3"
      break
    fi
    if [[ -x "${entry}/python" ]]; then
      path_python="${entry}/python"
      break
    fi
  done

  if [[ -n "${path_python}" ]]; then
    printf '%s\n' "${path_python}"
  elif [[ -n "${PREFIX:-}" && -x "${PREFIX}/bin/python3" ]]; then
    printf '%s\n' "${PREFIX}/bin/python3"
  elif [[ -x "/usr/bin/python3" ]]; then
    printf '%s\n' "/usr/bin/python3"
  elif [[ -x "/usr/bin/python" ]]; then
    printf '%s\n' "/usr/bin/python"
  fi
}

configure_android_build_env() {
  local detected_api=""
  local os_name=""
  local cargo_target_dir=""

  os_name="$(uname -o 2>/dev/null || true)"
  if [[ "${os_name}" != "Android" && -z "${TERMUX_VERSION:-}" ]]; then
    return 0
  fi

  if [[ -z "${ANDROID_API_LEVEL:-}" ]] && command -v getprop >/dev/null 2>&1; then
    detected_api="$(getprop ro.build.version.sdk 2>/dev/null || true)"
    if [[ -n "${detected_api}" ]]; then
      export ANDROID_API_LEVEL="${detected_api}"
    fi
  fi

  export ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-24}"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
  export MAX_JOBS="${MAX_JOBS:-1}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
  export NPY_NUM_BUILD_JOBS="${NPY_NUM_BUILD_JOBS:-1}"
  export NINJAFLAGS="${NINJAFLAGS:--j1}"
  cargo_target_dir="${CARGO_TARGET_DIR:-${FORGE_ROOT}/tmp/cargo-target}"
  mkdir -p "${cargo_target_dir}"
  export CARGO_TARGET_DIR="${cargo_target_dir}"
  echo "Android API level: ${ANDROID_API_LEVEL}"
  echo "Cargo build jobs: ${CARGO_BUILD_JOBS}"
  echo "Generic build jobs: ${MAX_JOBS}"
  echo "CMake parallel level: ${CMAKE_BUILD_PARALLEL_LEVEL}"
  echo "NumPy build jobs: ${NPY_NUM_BUILD_JOBS}"
  echo "Ninja flags: ${NINJAFLAGS}"
  echo "Cargo target dir: ${CARGO_TARGET_DIR}"
}

DEFAULT_PYTHON_BIN="$(resolve_host_python)"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
FORCE=0
WITH_REQUIREMENTS=1
SYSTEM_SITE=0

usage() {
  cat <<'EOF'
Usage: scripts/rebuild_eidosian_venv.sh [--force] [--without-requirements] [--system-site-packages]

Options:
  --force                 Rebuild even if venv already exists (backs up current venv first)
  --without-requirements  Skip dependency installation after creating venv
  --system-site-packages
                          Create venv with access to global site-packages
  -h, --help              Show help

Behavior:
  - Idempotent by default: if venv exists and --force is not set, script exits without changes.
  - Rollback-safe: when --force is used, existing venv is moved to Backups/eidosian_venv_<timestamp>.
  - Self-contained by default: installs requirements/eidosian_venv_reqs.txt when available.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --without-requirements)
      WITH_REQUIREMENTS=0
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

case "${PYTHON_BIN}" in
  "${VENV_DIR}/bin/"*)
    if [[ -n "${DEFAULT_PYTHON_BIN}" ]]; then
      echo "PYTHON_BIN points inside ${VENV_DIR}; falling back to host interpreter ${DEFAULT_PYTHON_BIN}" >&2
      PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
    fi
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Forge root: ${FORGE_ROOT}"
echo "Python: ${PYTHON_BIN}"
if [[ "${SYSTEM_SITE}" -eq 1 ]]; then
  echo "Mode: shared-site-packages (not self-contained)"
else
  echo "Mode: self-contained"
fi
configure_android_build_env

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
else
  echo "Dependency installation skipped (--without-requirements)."
fi

if [[ "${SYSTEM_SITE}" -ne 1 ]]; then
  os_name="$(uname -o 2>/dev/null || true)"
  if [[ "${os_name}" == "Android" || -n "${TERMUX_VERSION:-}" ]]; then
    if [[ -x "${FORGE_ROOT}/scripts/vendor_termux_python_packages.sh" ]]; then
      echo "Vendoring selected Termux native Python packages into ${VENV_DIR}"
      "${FORGE_ROOT}/scripts/vendor_termux_python_packages.sh"
    fi
  fi
fi

echo "Venv rebuild complete."
if [[ -n "${BACKUP_DIR}" ]]; then
  echo "Rollback command:"
  echo "  rm -rf '${VENV_DIR}' && mv '${BACKUP_DIR}' '${VENV_DIR}'"
fi
