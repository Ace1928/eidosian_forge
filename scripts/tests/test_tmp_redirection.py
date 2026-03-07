from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = FORGE_ROOT / "scripts" / "build_tmpredir.sh"
EXEC_SCRIPT = FORGE_ROOT / "scripts" / "eidos_exec.sh"
BOOTSTRAP = FORGE_ROOT / "shell" / "bootstrap.sh"
LIB_PATH = FORGE_ROOT / "build" / "libeidos_tmpredir.so"


def test_build_tmpredir_library() -> None:
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        return
    env = os.environ.copy()
    env["CC"] = compiler
    subprocess.run([str(BUILD_SCRIPT)], check=True, cwd=FORGE_ROOT, env=env)
    assert LIB_PATH.exists()


def test_eidos_exec_redirects_hardcoded_tmp(tmp_path: Path) -> None:
    if os.environ.get("TERMUX_VERSION") or "com.termux" in os.environ.get("PREFIX", ""):
        return
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        return
    env = os.environ.copy()
    env["CC"] = compiler
    subprocess.run([str(BUILD_SCRIPT)], check=True, cwd=FORGE_ROOT, env=env)

    target_tmp = tmp_path / "tmp-root"
    script = "import pathlib; p=pathlib.Path('/tmp/eidos-hardcoded-test.txt'); p.write_text('ok', encoding='utf-8'); print(p.exists())"
    run_env = env.copy()
    run_env["EIDOS_TMPDIR"] = str(target_tmp)
    run_env["EIDOS_ENABLE_TMP_PRELOAD"] = "1"
    subprocess.run([str(EXEC_SCRIPT), "python3", "-c", script], check=True, cwd=FORGE_ROOT, env=run_env)
    assert (target_tmp / "eidos-hardcoded-test.txt").read_text(encoding="utf-8") == "ok"


def test_bootstrap_exports_tmpdir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    bashrc = home / ".bashrc"
    bashrc.write_text(
        "#!/usr/bin/env bash\n"
        f"source {BOOTSTRAP}\n"
        "printf 'TMPDIR=%s\\n' \"${TMPDIR}\"\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "EIDOS_ENABLE_DOC_FORGE_AUTOSTART": "0",
            "EIDOS_ENABLE_ATLAS_AUTOSTART": "0",
            "EIDOS_ENABLE_SCHEDULER_AUTOSTART": "0",
            "EIDOS_ENABLE_OLLAMA_AUTOSTART": "0",
            "EIDOS_ENABLE_X11_AUTOSTART": "0",
            "EIDOS_ENABLE_PULSEAUDIO_AUTOSTART": "0",
            "EIDOS_ENABLE_FILES_DASHBOARD_AUTOSTART": "0",
            "EIDOS_DISABLE_NOTIFICATIONS": "1",
            "PREFIX": "/data/data/com.termux/files/usr",
            "TERMUX_VERSION": "1.0",
        }
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--rcfile", str(bashrc), "-ic", "true"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "TMPDIR=/data/data/com.termux/files/usr/tmp/eidos-" in result.stdout


def test_eidos_exec_sets_tmp_contract_without_preload(tmp_path: Path) -> None:
    target_tmp = tmp_path / "tmp-root"
    env = os.environ.copy()
    env["EIDOS_TMPDIR"] = str(target_tmp)
    result = subprocess.run(
        [str(EXEC_SCRIPT), "python3", "-c", "import os; print(os.environ['TMPDIR']); print(os.environ.get('LD_PRELOAD',''))"],
        check=True,
        cwd=FORGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines()]
    assert lines[0] == str(target_tmp)
    assert "libeidos_tmpredir.so" not in lines[1]
