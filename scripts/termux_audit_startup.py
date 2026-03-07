#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", HOME / "eidosian_forge")).resolve()
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
REPORT_PATH = RUNTIME_DIR / "termux_startup_audit.json"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _contains(text: str, needle: str) -> bool:
    return needle in text if text else False


def main() -> int:
    bashrc = HOME / ".bashrc"
    termux_props = HOME / ".termux" / "termux.properties"
    boot_dir = HOME / ".termux" / "boot"
    venv_bin = FORGE_ROOT / "eidosian_venv" / "bin"
    path_entries = os.environ.get("PATH", "").split(":")
    qwenchat_hit = shutil.which("qwenchat")

    bashrc_text = _read(bashrc)
    props_text = _read(termux_props)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "home": str(HOME),
        "forge_root": str(FORGE_ROOT),
        "paths": {
            "bashrc": str(bashrc),
            "termux_properties": str(termux_props),
            "boot_dir": str(boot_dir),
            "venv_bin": str(venv_bin),
        },
        "venv": {
            "activate_exists": (venv_bin / "activate").exists(),
            "python_exists": (venv_bin / "python").exists(),
            "pip_exists": (venv_bin / "pip").exists() or (venv_bin / "pip3.13").exists(),
        },
        "path": {
            "entries": path_entries,
            "scripts_precedes_forge": (
                str(HOME / "scripts") in path_entries
                and str(venv_bin) in path_entries
                and path_entries.index(str(HOME / "scripts")) < path_entries.index(str(venv_bin))
            ),
            "qwenchat_resolved": qwenchat_hit or "",
        },
        "termux": {
            "allow_external_apps_enabled": _contains(props_text, "allow-external-apps = true"),
            "termux_properties_exists": termux_props.exists(),
            "boot_dir_exists": boot_dir.exists(),
            "boot_scripts": sorted(str(p) for p in boot_dir.glob("*")) if boot_dir.exists() else [],
        },
        "bashrc": {
            "exists": bashrc.exists(),
            "line_count": len(bashrc_text.splitlines()) if bashrc_text else 0,
            "sources_eidos_env": _contains(bashrc_text, "eidos_env.sh"),
            "sources_shell_bootstrap": _contains(bashrc_text, "shell/bootstrap.sh"),
            "sources_termux_bootstrap": _contains(bashrc_text, "termux_bootstrap.sh"),
            "starts_eidos_services": _contains(bashrc_text, "eidos_termux_services.sh"),
            "starts_x11": _contains(bashrc_text, "start_x11"),
            "contains_termux_build_flags": _contains(bashrc_text, "BLIS_ARCH"),
            "contains_files_dashboard": _contains(bashrc_text, "FILES_DASHBOARD_DIR"),
        },
        "shell_modules": {
            "termux_bootstrap_exists": (FORGE_ROOT / "shell" / "termux_bootstrap.sh").exists(),
            "shell_bootstrap_exists": (FORGE_ROOT / "shell" / "bootstrap.sh").exists(),
            "profile_modules": (
                sorted(str(p.relative_to(FORGE_ROOT)) for p in (FORGE_ROOT / "shell" / "profile.d").glob("*.sh"))
                if (FORGE_ROOT / "shell" / "profile.d").exists()
                else []
            ),
        },
        "recommendations": [],
    }

    recs = report["recommendations"]
    if not report["venv"]["activate_exists"]:
        recs.append("restore venv activation scripts")
    if report["path"]["scripts_precedes_forge"]:
        recs.append("move forge wrappers ahead of ~/scripts in PATH")
    if not report["termux"]["boot_dir_exists"]:
        recs.append("create ~/.termux/boot for boot-time orchestration")
    if not report["termux"]["allow_external_apps_enabled"]:
        recs.append("consider allow-external-apps = true if RUN_COMMAND integration is desired")
    if report["bashrc"]["line_count"] > 250:
        recs.append("thin ~/.bashrc by moving runtime logic into sourced modules")
    if not report["bashrc"]["sources_shell_bootstrap"]:
        recs.append("source shell/bootstrap.sh from ~/.bashrc")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(str(REPORT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
