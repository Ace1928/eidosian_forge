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



def test_eidos_termux_services_pause_resume_runit_command(tmp_path: Path) -> None:
    home = tmp_path / "home"
    prefix = tmp_path / "usr"
    service_root = prefix / "var" / "service"
    (prefix / "bin").mkdir(parents=True)
    service_root.mkdir(parents=True)
    (home / ".eidosian" / "run").mkdir(parents=True)
    sv = prefix / "bin" / "sv"
    sv.write_text(
        "#!/bin/sh\nprintf '%s %s\\n' \"$1\" \"$2\" >> \"${HOME}/sv.log\"\nexit 0\n",
        encoding="utf-8",
    )
    sv.chmod(0o755)
    (service_root / "eidos-scheduler").mkdir(parents=True)

    env = os.environ.copy()
    env.update({"HOME": str(home), "PREFIX": str(prefix), "EIDOS_FORGE_ROOT": str(FORGE_ROOT), "PATH": str(prefix / "bin") + ":" + env.get("PATH", ""), "EIDOS_RUNIT_SERVICE_DIR": str(service_root)})
    subprocess.run([str(SERVICES), "pause", "scheduler"], check=True, env=env, cwd=FORGE_ROOT)
    subprocess.run([str(SERVICES), "resume", "scheduler"], check=True, env=env, cwd=FORGE_ROOT)

    log = (home / "sv.log").read_text(encoding="utf-8")
    assert "pause" in log
    assert "cont" in log


def test_eidos_termux_services_pause_resume_pid_fallback(tmp_path: Path) -> None:
    home = tmp_path / "home"
    forge = tmp_path / "forge"
    run_dir = home / ".eidosian" / "run"
    atlas_script = forge / "web_interface_forge" / "scripts" / "run_dashboard.sh"
    atlas_script.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    atlas_script.write_text('#!/bin/sh\nsleep 60\n', encoding="utf-8")
    atlas_script.chmod(0o755)
    proc = subprocess.Popen(["bash", str(atlas_script)], cwd=str(forge))
    try:
        (run_dir / "eidos_atlas.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        env = os.environ.copy()
        env.update({"HOME": str(home), "EIDOS_FORGE_ROOT": str(forge), "EIDOS_RUNIT_SERVICE_DIR": str(forge / "no-runit-services")})
        subprocess.run([str(SERVICES), "pause", "atlas"], check=True, env=env, cwd=FORGE_ROOT)
        paused = subprocess.run(
            [str(SERVICES), "status", "atlas"],
            check=True,
            env=env,
            cwd=FORGE_ROOT,
            capture_output=True,
            text=True,
        )
        assert "paused(managed pid=" in paused.stdout
        subprocess.run([str(SERVICES), "resume", "atlas"], check=True, env=env, cwd=FORGE_ROOT)
        resumed = subprocess.run(
            [str(SERVICES), "status", "atlas"],
            check=True,
            env=env,
            cwd=FORGE_ROOT,
            capture_output=True,
            text=True,
        )
        assert "running(managed pid=" in resumed.stdout
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)



def test_eidos_termux_services_low_load_profile(tmp_path: Path) -> None:
    home = tmp_path / "home"
    prefix = tmp_path / "usr"
    service_root = prefix / "var" / "service"
    (prefix / "bin").mkdir(parents=True)
    service_root.mkdir(parents=True)
    (home / ".eidosian" / "run").mkdir(parents=True)
    sv = prefix / "bin" / "sv"
    sv.write_text(
        "#!/bin/sh\nprintf '%s %s\n' \"$1\" \"$2\" >> \"${HOME}/sv.log\"\nexit 0\n",
        encoding="utf-8",
    )
    sv.chmod(0o755)
    for name in (
        "eidos-mcp",
        "eidos-atlas",
        "eidos-scheduler",
        "eidos-local-agent",
        "eidos-ollama-qwen",
        "eidos-ollama-embedding",
        "eidos-doc-forge",
    ):
        (service_root / name).mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "PREFIX": str(prefix),
        "EIDOS_FORGE_ROOT": str(FORGE_ROOT),
        "PATH": str(prefix / "bin") + ":" + env.get("PATH", ""),
        "EIDOS_RUNIT_SERVICE_DIR": str(service_root),
    })
    subprocess.run([str(SERVICES), "low-load"], check=True, env=env, cwd=FORGE_ROOT)

    log = (home / "sv.log").read_text(encoding="utf-8")
    assert "up " in log
    assert "pause " in log
    assert "down " in log
