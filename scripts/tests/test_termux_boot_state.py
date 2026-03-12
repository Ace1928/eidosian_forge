from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
BOOT_SCRIPT = FORGE_ROOT / "scripts" / "eidos_termux_boot.sh"


def test_termux_boot_writes_state_and_installs_runit(tmp_path: Path) -> None:
    home = tmp_path / "home"
    forge = tmp_path / "forge"
    prefix = tmp_path / "usr"
    (home / ".eidosian" / "run").mkdir(parents=True, exist_ok=True)
    (forge / "scripts").mkdir(parents=True, exist_ok=True)
    (forge / "data" / "runtime").mkdir(parents=True, exist_ok=True)
    (prefix / "etc" / "profile.d").mkdir(parents=True, exist_ok=True)
    (prefix / "var" / "service").mkdir(parents=True, exist_ok=True)

    (forge / "scripts" / "eidos_termux_services.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (forge / "scripts" / "install_termux_runit_services.sh").write_text("#!/bin/sh\nmkdir -p \"$PREFIX/var/service/eidos-scheduler\"\n", encoding="utf-8")
    (forge / "scripts" / "write_runtime_capabilities.py").write_text("print('{}')\n", encoding="utf-8")
    (forge / "eidosian_venv" / "bin").mkdir(parents=True, exist_ok=True)
    (forge / "eidosian_venv" / "bin" / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (prefix / "etc" / "profile.d" / "start-services.sh").write_text("#!/bin/sh\n:\n", encoding="utf-8")

    for path in (
        forge / "scripts" / "eidos_termux_services.sh",
        forge / "scripts" / "install_termux_runit_services.sh",
        forge / "eidosian_venv" / "bin" / "python",
    ):
        path.chmod(0o755)

    env = os.environ.copy()
    env.update({"HOME": str(home), "PREFIX": str(prefix), "EIDOS_FORGE_ROOT": str(forge)})
    subprocess.run([str(BOOT_SCRIPT)], check=True, env=env, cwd=FORGE_ROOT)

    state = json.loads((forge / "data" / "runtime" / "termux_boot_status.json").read_text(encoding="utf-8"))
    assert state["contract"] == "eidos.termux_boot_status.v1"
    assert state["services_script_present"] is True
    assert (prefix / "var" / "service" / "eidos-scheduler").exists()
