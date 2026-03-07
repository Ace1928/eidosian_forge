from __future__ import annotations

import os
import subprocess
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
INSTALL_RUNIT = FORGE_ROOT / "scripts" / "install_termux_runit_services.sh"
SERVICES = FORGE_ROOT / "scripts" / "eidos_termux_services.sh"


def test_install_termux_runit_services_creates_service_tree(tmp_path: Path) -> None:
    home = tmp_path / "home"
    prefix = tmp_path / "usr"
    service_root = prefix / "var" / "service"
    svlogger = prefix / "share" / "termux-services" / "svlogger"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "var" / "service").mkdir(parents=True)
    (prefix / "share" / "termux-services").mkdir(parents=True)
    (home / ".eidosian" / "log" / "sv").mkdir(parents=True)
    for name in ("bash", "sh"):
        target = prefix / "bin" / name
        target.write_text('#!/bin/sh\nexec /bin/sh "$@"\n', encoding="utf-8")
        target.chmod(0o755)
    svlogger.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    svlogger.chmod(0o755)

    env = os.environ.copy()
    env.update({"HOME": str(home), "PREFIX": str(prefix), "EIDOS_FORGE_ROOT": str(FORGE_ROOT)})
    subprocess.run([str(INSTALL_RUNIT)], check=True, env=env, cwd=FORGE_ROOT)

    scheduler = service_root / "eidos-scheduler"
    assert scheduler.exists()
    assert (scheduler / "run").exists()
    assert (scheduler / "log" / "run").exists()
    assert (scheduler / "down").exists()
    run_text = (scheduler / "run").read_text(encoding="utf-8")
    assert "run_eidos_scheduler.sh" in run_text
    assert 'EIDOS_SERVICE_SUPERVISION="runit"' in run_text


def test_eidos_termux_services_install_runit_command(tmp_path: Path) -> None:
    home = tmp_path / "home"
    prefix = tmp_path / "usr"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "var" / "service").mkdir(parents=True)
    (prefix / "share" / "termux-services").mkdir(parents=True)
    for name in ("bash", "sh"):
        target = prefix / "bin" / name
        target.write_text('#!/bin/sh\nexec /bin/sh "$@"\n', encoding="utf-8")
        target.chmod(0o755)
    svlogger = prefix / "share" / "termux-services" / "svlogger"
    svlogger.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    svlogger.chmod(0o755)

    env = os.environ.copy()
    env.update({"HOME": str(home), "PREFIX": str(prefix), "EIDOS_FORGE_ROOT": str(FORGE_ROOT)})
    subprocess.run([str(SERVICES), "install-runit"], check=True, env=env, cwd=FORGE_ROOT)

    assert (prefix / "var" / "service" / "eidos-mcp" / "run").exists()
