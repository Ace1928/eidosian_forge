#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${FORGE_ROOT}/eidosian_venv}"
HOST_PYTHON="${HOST_PYTHON:-/data/data/com.termux/files/usr/bin/python}"

if [[ $# -gt 0 ]]; then
  PACKAGES=("$@")
else
  read -r -a PACKAGES <<<"${TERMUX_VENDOR_PY_PKGS:-numpy}"
fi

if [[ ! -x "${HOST_PYTHON}" ]]; then
  echo "Host Termux python not found at ${HOST_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Venv python not found at ${VENV_DIR}/bin/python" >&2
  exit 1
fi

if [[ "${#PACKAGES[@]}" -eq 0 ]]; then
  echo "No packages requested for vendoring."
  exit 0
fi

PACKAGE_LIST="$(printf '%s\n' "${PACKAGES[@]}")"

HOST_JSON="$(
  env -i \
    PATH=/data/data/com.termux/files/usr/bin \
    HOME="${HOME}" \
    TERM="${TERM:-xterm-256color}" \
    "${HOST_PYTHON}" - <<'PY' "${PACKAGE_LIST}"
import importlib
import importlib.metadata as metadata
import json
import pathlib
import sys

packages = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
result = {"packages": {}, "errors": {}}

for package in packages:
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        result["errors"][package] = f"import failed: {exc}"
        continue

    entries = []
    module_path = pathlib.Path(getattr(module, "__file__", "")).resolve()
    if module_path.name == "__init__.py":
        entries.append(str(module_path.parent))
    elif module_path.exists():
        entries.append(str(module_path))

    dist_paths = []
    for dist in metadata.distributions():
        try:
            name = (dist.metadata.get("Name") or "").strip()
        except Exception:
            continue
        if not name:
            continue
        normalized = name.lower().replace("-", "_")
        if normalized != package.lower().replace("-", "_"):
            continue
        dist_path = getattr(dist, "_path", None)
        if dist_path is not None:
            dist_paths.append(str(pathlib.Path(dist_path).resolve()))

    result["packages"][package] = {
        "module_entries": entries,
        "dist_entries": sorted(set(dist_paths)),
    }

print(json.dumps(result))
PY
)"

VENV_SITE_PACKAGES="$("${VENV_DIR}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

HOST_JSON="${HOST_JSON}" VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES}" "${VENV_DIR}/bin/python" - <<'PY'
import json
import os
import pathlib
import shutil
import sys

payload = json.loads(os.environ["HOST_JSON"])
target_root = pathlib.Path(os.environ["VENV_SITE_PACKAGES"]).resolve()
target_root.mkdir(parents=True, exist_ok=True)

errors = payload.get("errors", {})
if errors:
    for package, message in errors.items():
        print(f"Skipping {package}: {message}", file=sys.stderr)

for package, info in payload.get("packages", {}).items():
    for source in info.get("module_entries", []) + info.get("dist_entries", []):
        src = pathlib.Path(source)
        if not src.exists():
            continue
        dest = target_root / src.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if src.is_dir():
            shutil.copytree(src, dest, symlinks=True)
        else:
            shutil.copy2(src, dest)
        print(f"Vendored {package}: {src} -> {dest}")
PY
