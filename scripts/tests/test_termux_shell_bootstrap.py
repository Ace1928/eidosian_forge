from __future__ import annotations

import os
import subprocess
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SHELL = FORGE_ROOT / "scripts" / "install_shell_bootstrap.sh"
INSTALL_BOOT = FORGE_ROOT / "scripts" / "install_termux_boot.sh"
INSTALL_RUNIT = FORGE_ROOT / "scripts" / "install_termux_runit_services.sh"
AUDIT = FORGE_ROOT / "scripts" / "termux_audit_startup.py"
BOOTSTRAP = FORGE_ROOT / "shell" / "bootstrap.sh"


def test_install_shell_bootstrap_writes_thin_bashrc(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text("echo legacy\n", encoding="utf-8")
    env = os.environ.copy()
    env["HOME"] = str(home)
    subprocess.run([str(INSTALL_SHELL)], check=True, env=env, cwd=FORGE_ROOT)
    text = (home / ".bashrc").read_text(encoding="utf-8")
    assert "shell/bootstrap.sh" in text
    backups = list((FORGE_ROOT / "backups" / "shell_bootstrap").glob("bashrc.*.bak"))
    assert backups


def test_install_termux_boot_writes_wrapper(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    env["EIDOS_INSTALL_RUNIT_ON_BOOT_INSTALL"] = "0"
    subprocess.run([str(INSTALL_BOOT)], check=True, env=env, cwd=FORGE_ROOT)
    wrapper = home / ".termux" / "boot" / "00-eidos-boot"
    assert wrapper.exists()
    text = wrapper.read_text(encoding="utf-8")
    assert "eidos_termux_boot.sh" in text


def test_install_termux_runit_script_exists() -> None:
    assert INSTALL_RUNIT.exists()


def test_bootstrap_shell_smoke(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    bashrc = home / ".bashrc"
    bashrc.write_text(
        "#!/usr/bin/env bash\n"
        f"source {BOOTSTRAP}\n"
        "printf 'BOOTSTRAP_OK=%s\\n' \"${EIDOS_SHELL_COMMON_HELPERS_LOADED:-0}\"\n"
        "printf 'HAS_FORGE_SCRIPTS=%s\\n' \"$(printf '%s' \"${PATH}\" | grep -Fq '/eidosian_forge/scripts' && echo 1 || echo 0)\"\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "EIDOS_FORGE_ROOT": str(FORGE_ROOT),
            "EIDOS_ENABLE_DOC_FORGE_AUTOSTART": "0",
            "EIDOS_ENABLE_ATLAS_AUTOSTART": "0",
            "EIDOS_ENABLE_SCHEDULER_AUTOSTART": "0",
            "EIDOS_ENABLE_OLLAMA_AUTOSTART": "0",
            "EIDOS_ENABLE_X11_AUTOSTART": "0",
            "EIDOS_ENABLE_PULSEAUDIO_AUTOSTART": "0",
            "EIDOS_ENABLE_FILES_DASHBOARD_AUTOSTART": "0",
            "EIDOS_DISABLE_NOTIFICATIONS": "1",
            "PREFIX": "/usr",
        }
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--rcfile", str(bashrc), "-ic", "true"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "BOOTSTRAP_OK=1" in result.stdout
    assert "HAS_FORGE_SCRIPTS=1" in result.stdout


def test_termux_audit_detects_shell_bootstrap(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(f"source {BOOTSTRAP}\n", encoding="utf-8")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["EIDOS_FORGE_ROOT"] = str(FORGE_ROOT)
    subprocess.run([str(AUDIT)], check=True, env=env, cwd=FORGE_ROOT)
    report = (FORGE_ROOT / "data" / "runtime" / "termux_startup_audit.json").read_text(encoding="utf-8")
    assert '"sources_shell_bootstrap": true' in report
