from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def log_info(message: str) -> None:
    print(f"INFO: {message}", file=sys.stdout, flush=True)


def log_warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stdout, flush=True)


def log_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)


def die(message: str, code: int = 1) -> None:
    log_error(message)
    raise SystemExit(int(code))


def load_env_file(path: Optional[PathLike]) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = value.strip()


def read_env_config(name: str, base_dir: Optional[PathLike] = None) -> Path:
    base = Path(base_dir).expanduser() if base_dir else Path.cwd()
    return (base / f"{name}.env").resolve()


def ensure_parent_dir(path: PathLike) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def normalize_exit_code(code: int) -> int:
    return 0 if int(code) == 0 else 1


def require_cmd(command: str) -> str:
    resolved = shutil.which(command)
    if not resolved:
        die(f"Required command not found: {command}", code=1)
    return resolved


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
