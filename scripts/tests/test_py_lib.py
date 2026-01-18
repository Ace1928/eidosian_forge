import os
import shutil
from pathlib import Path

from lib import py_lib


def test_loggers(capsys):
    py_lib.log_info("hello")
    py_lib.log_warn("warn")
    py_lib.log_error("err")
    out = capsys.readouterr()
    assert "INFO: hello" in out.out
    assert "WARN: warn" in out.out
    assert "ERROR: err" in out.err


def test_die(capsys):
    try:
        py_lib.die("nope", code=2)
    except SystemExit as exc:
        assert exc.code == 2
    out = capsys.readouterr()
    assert "ERROR: nope" in out.err


def test_load_env_file(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("A=1\n#comment\nB=two\nINVALID\n")
    py_lib.load_env_file(env_file)
    assert os.environ.get("A") == "1"
    assert os.environ.get("B") == "two"


def test_load_env_file_none():
    py_lib.load_env_file(None)


def test_load_env_file_missing(tmp_path):
    missing = tmp_path / "missing.env"
    py_lib.load_env_file(missing)


def test_read_env_config():
    path = py_lib.read_env_config("example")
    assert path.name == "example.env"


def test_ensure_parent_dir(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    py_lib.ensure_parent_dir(target)
    assert target.parent.exists()


def test_normalize_exit_code():
    assert py_lib.normalize_exit_code(0) == 0
    assert py_lib.normalize_exit_code(2) == 1


def test_require_cmd_failure(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    try:
        py_lib.require_cmd("missing")
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit")


def test_require_cmd_success(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/bin/echo")
    py_lib.require_cmd("echo")
