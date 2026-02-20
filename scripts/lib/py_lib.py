from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT_PY_LIB = Path(__file__).resolve().parents[2] / "lib" / "py_lib.py"
_SPEC = importlib.util.spec_from_file_location("eidos_root_py_lib", _ROOT_PY_LIB)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - hard failure path
    raise RuntimeError(f"Unable to load root helper module: {_ROOT_PY_LIB}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


log_info = _MODULE.log_info
log_warn = _MODULE.log_warn
log_error = _MODULE.log_error
die = _MODULE.die
load_env_file = _MODULE.load_env_file
read_env_config = _MODULE.read_env_config
ensure_parent_dir = _MODULE.ensure_parent_dir
normalize_exit_code = _MODULE.normalize_exit_code
require_cmd = _MODULE.require_cmd


__all__ = [
    "die",
    "ensure_parent_dir",
    "load_env_file",
    "log_error",
    "log_info",
    "log_warn",
    "normalize_exit_code",
    "read_env_config",
    "require_cmd",
]

