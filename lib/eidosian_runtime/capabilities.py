from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _forge_root() -> Path:
    raw = str(os.environ.get("EIDOS_FORGE_ROOT") or os.environ.get("EIDOS_FORGE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _is_termux() -> bool:
    prefix = str(os.environ.get("PREFIX") or "")
    return "com.termux" in prefix or bool(os.environ.get("TERMUX_VERSION"))


@dataclass(slots=True)
class RuntimeCapabilities:
    contract: str
    platform: str
    system: str
    python_version: str
    forge_root: str
    home_root: str
    prefix_root: str
    tmpdir: str
    runtime_dir: str
    runit_service_root: str
    termux_boot_root: str
    has_termux_api: bool
    has_termux_boot: bool
    has_termux_wake_lock: bool
    has_termux_open_url: bool
    has_sv: bool
    has_runsvdir: bool
    has_ollama: bool
    has_uvicorn: bool
    has_x11_launcher: bool
    has_notification: bool
    default_dashboard_url: str


def collect_runtime_capabilities() -> RuntimeCapabilities:
    forge_root = _forge_root()
    home_root = Path(os.environ.get("HOME", str(forge_root.parent))).expanduser().resolve()
    prefix_root = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr")).expanduser()
    runtime_dir = forge_root / "data" / "runtime"
    runit_service_root = Path(
        os.environ.get("EIDOS_RUNIT_SERVICE_DIR", str(prefix_root / "var" / "service"))
    ).expanduser()
    termux_boot_root = home_root / ".termux" / "boot"
    dashboard_port = str(os.environ.get("EIDOS_ATLAS_PORT") or 8936)
    return RuntimeCapabilities(
        contract="eidos.runtime_capabilities.v1",
        platform="termux" if _is_termux() else "linux",
        system=platform.system().lower(),
        python_version=platform.python_version(),
        forge_root=str(forge_root),
        home_root=str(home_root),
        prefix_root=str(prefix_root),
        tmpdir=str(Path(os.environ.get("TMPDIR", str(prefix_root / "tmp")))),
        runtime_dir=str(runtime_dir),
        runit_service_root=str(runit_service_root),
        termux_boot_root=str(termux_boot_root),
        has_termux_api=shutil.which("termux-info") is not None,
        has_termux_boot=termux_boot_root.exists(),
        has_termux_wake_lock=shutil.which("termux-wake-lock") is not None,
        has_termux_open_url=shutil.which("termux-open-url") is not None,
        has_sv=shutil.which("sv") is not None,
        has_runsvdir=shutil.which("runsvdir") is not None,
        has_ollama=shutil.which("ollama") is not None,
        has_uvicorn=shutil.which("uvicorn") is not None,
        has_x11_launcher=(home_root / "scripts" / "start_x11").exists(),
        has_notification=(shutil.which("termux-notification") is not None or shutil.which("notify-send") is not None),
        default_dashboard_url=f"http://127.0.0.1:{dashboard_port}",
    )


def write_runtime_capabilities(output_path: str | Path | None = None) -> dict[str, Any]:
    payload = asdict(collect_runtime_capabilities())
    path = Path(output_path) if output_path else _forge_root() / "data" / "runtime" / "platform_capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
