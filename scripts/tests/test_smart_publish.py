import importlib.machinery
import importlib.util
import importlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SMART_PUBLISH_PATH = Path(__file__).resolve().parents[1] / "smart_publish"


def load_smart_publish():
    loader = importlib.machinery.SourceFileLoader(
        "smart_publish", str(SMART_PUBLISH_PATH)
    )
    spec = importlib.util.spec_from_loader("smart_publish", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


smart_publish = load_smart_publish()


def test_parse_version():
    assert smart_publish.parse_version("1.2.3") == (1, 2, 3)
    with pytest.raises(ValueError):
        smart_publish.parse_version("bad")


def test_increment_version():
    assert (
        smart_publish.increment_version("1.2.3", smart_publish.VersionBump.PATCH)
        == "1.2.4"
    )
    assert (
        smart_publish.increment_version("1.2.3", smart_publish.VersionBump.MINOR)
        == "1.3.0"
    )
    assert (
        smart_publish.increment_version("1.2.3", smart_publish.VersionBump.MAJOR)
        == "2.0.0"
    )
    assert (
        smart_publish.increment_version("1.2.3", smart_publish.VersionBump.NONE)
        == "1.2.3"
    )


def test_find_package_root(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    search_base = tmp_path / "search"
    search_base.mkdir()
    (search_base / "pkg").mkdir()
    cwd = Path.cwd()
    try:
        os.chdir(pkg)
        assert smart_publish.find_package_root("pkg") == pkg
    finally:
        os.chdir(cwd)

    try:
        os.chdir(tmp_path)
        assert smart_publish.find_package_root("pkg") == pkg
        assert smart_publish.find_package_root("missing") is None
        assert smart_publish.find_package_root("pkg", [tmp_path]) == pkg
        assert smart_publish.find_package_root("missing", [empty]) is None
    finally:
        os.chdir(cwd)

    try:
        os.chdir(empty)
        assert (
            smart_publish.find_package_root("pkg", [search_base]) == search_base / "pkg"
        )
        assert smart_publish.find_package_root("missing", [tmp_path / "nope"]) is None
    finally:
        os.chdir(cwd)


def test_update_version_in_file(tmp_path):
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text("version = 1.2.3\n")
    assert smart_publish.update_version_in_file(setup_cfg, "1.2.3", "1.2.4")
    assert "1.2.4" in setup_cfg.read_text()

    init_py = tmp_path / "__init__.py"
    init_py.write_text("__version__ = '1.2.3'\n")
    assert smart_publish.update_version_in_file(init_py, "1.2.3", "1.2.4")

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "1.2.3"\n')
    assert smart_publish.update_version_in_file(pyproject, "1.2.3", "1.2.4")

    setup_cfg.write_text("version = 1.2.4\n")
    assert smart_publish.update_version_in_file(setup_cfg, "1.2.4", "1.2.4") is False

    other = tmp_path / "other.txt"
    other.write_text("noop")
    assert smart_publish.update_version_in_file(other, "1.2.3", "1.2.4") is False

    missing = tmp_path / "missing.cfg"
    assert smart_publish.update_version_in_file(missing, "1.2.3", "1.2.4") is False


def test_update_package_version(tmp_path):
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    nested_dir = tmp_path / tmp_path.name
    nested_dir.mkdir()
    (nested_dir / "__init__.py").write_text("__version__ = '1.2.3'\n")
    updated = smart_publish.update_package_version(tmp_path, "1.2.3", "1.2.4")
    assert updated >= 1


def test_update_package_version_no_nested(tmp_path):
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    updated = smart_publish.update_package_version(tmp_path, "1.2.3", "1.2.4")
    assert updated >= 1


def test_get_current_version(tmp_path):
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text("version = 1.2.3\n")
    assert smart_publish.get_current_version(tmp_path) == "1.2.3"

    setup_cfg.write_text("no version here\n")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "2.0.0"\n')
    assert smart_publish.get_current_version(tmp_path) == "2.0.0"

    pyproject.unlink()
    init_py = tmp_path / "__init__.py"
    init_py.write_text("__version__ = '3.0.0'\n")
    assert smart_publish.get_current_version(tmp_path) == "3.0.0"

    init_py.write_text("no version here\n")
    assert smart_publish.get_current_version(tmp_path) is None


def test_get_current_version_no_match(tmp_path):
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text("no version here\n")
    assert smart_publish.get_current_version(tmp_path) is None


def test_load_pyproject_version_without_tomllib(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "1.0.0"\n')
    monkeypatch.setattr(smart_publish, "tomllib", None)
    assert smart_publish.load_pyproject_version(pyproject) is None


def test_load_pyproject_version_bad_toml(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("bad = [")
    assert smart_publish.load_pyproject_version(pyproject) is None


def test_load_pyproject_version_non_string(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nversion = 1\n")
    assert smart_publish.load_pyproject_version(pyproject) is None


def test_build_wheel_for_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.build_wheel_for_platform(tmp_path, "linux-x86_64") == 0
    assert smart_publish.build_wheel_for_platform(tmp_path, "unknown") == 1


def test_build_multi_platform_wheels(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "file.whl").write_text("x")

    monkeypatch.setattr(
        smart_publish, "build_wheel_for_platform", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.build_multi_platform_wheels(tmp_path, ["all"]) is True
    assert smart_publish.build_multi_platform_wheels(tmp_path, ["unknown"]) is True

    monkeypatch.setattr(
        smart_publish, "build_wheel_for_platform", lambda *args, **kwargs: 1
    )
    assert (
        smart_publish.build_multi_platform_wheels(tmp_path, ["linux-x86_64"]) is False
    )

    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 1)
    assert (
        smart_publish.build_multi_platform_wheels(tmp_path, ["linux-x86_64"]) is False
    )


def test_build_multi_platform_wheels_no_dist(monkeypatch, tmp_path):
    monkeypatch.setattr(
        smart_publish, "build_wheel_for_platform", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.build_multi_platform_wheels(tmp_path, ["linux-x86_64"]) is True


def test_run_publish_command_with_publish_script(monkeypatch, tmp_path):
    publish = tmp_path / "publish.py"
    publish.write_text("print('ok')\n")
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.run_publish_command(tmp_path, ["--skip-existing"]) == 0


def test_run_publish_command_with_platforms(monkeypatch, tmp_path):
    monkeypatch.setattr(
        smart_publish, "build_multi_platform_wheels", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert (
        smart_publish.run_publish_command(
            tmp_path, ["--skip-existing"], platforms=["linux-x86_64"]
        )
        == 0
    )

    monkeypatch.setattr(
        smart_publish, "build_multi_platform_wheels", lambda *args, **kwargs: False
    )
    assert (
        smart_publish.run_publish_command(tmp_path, [], platforms=["linux-x86_64"]) == 1
    )


def test_run_publish_command_build_fail(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    monkeypatch.setattr(smart_publish, "build_distributions", lambda *args, **kwargs: 1)
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.run_publish_command(tmp_path, []) == 1


def test_run_publish_command_build_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(smart_publish, "build_distributions", lambda *args, **kwargs: 0)
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    assert smart_publish.run_publish_command(tmp_path, []) == 0


def test_setup_auth_env(monkeypatch):
    monkeypatch.setenv("PYPI_TOKEN", "token")
    args = smart_publish.setup_auth(None)
    assert "__token__" in args


def test_setup_auth_pypirc(tmp_path, monkeypatch):
    monkeypatch.delenv("PYPI_TOKEN", raising=False)
    pypirc = tmp_path / "pypirc"
    pypirc.write_text("[distutils]\n")
    args = smart_publish.setup_auth(str(pypirc))
    assert "--config-file" in args


def test_setup_auth_missing_pypirc(monkeypatch, tmp_path):
    monkeypatch.delenv("PYPI_TOKEN", raising=False)
    assert smart_publish.setup_auth(str(tmp_path / "missing")) == []


def test_setup_auth_default(monkeypatch):
    monkeypatch.delenv("PYPI_TOKEN", raising=False)
    assert smart_publish.setup_auth(None) == []


def test_build_distributions(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)
    assert smart_publish.build_distributions(tmp_path) == 0

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert smart_publish.build_distributions(tmp_path) == 0

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert smart_publish.build_distributions(tmp_path) == 1


def test_validate_version():
    smart_publish.validate_version("1.2.3")
    with pytest.raises(ValueError):
        smart_publish.validate_version("bad.version")


def test_main_errors(monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(smart_publish.sys, "argv", ["smart_publish", "missing"])
    assert smart_publish.main() == 1


def test_main_no_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    monkeypatch.setattr(
        smart_publish, "get_current_version", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(smart_publish.sys, "argv", ["smart_publish", "pkg"])
    assert smart_publish.main() == 1


def test_main_invalid_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    monkeypatch.setattr(
        smart_publish, "get_current_version", lambda *args, **kwargs: "1.2.3"
    )
    monkeypatch.setattr(
        smart_publish,
        "validate_version",
        lambda v: (_ for _ in ()).throw(ValueError("bad")),
    )
    monkeypatch.setattr(
        smart_publish.sys, "argv", ["smart_publish", "pkg", "--version", "bad"]
    )
    assert smart_publish.main() == 1


def test_main_dry_run_no_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(
        smart_publish.sys,
        "argv",
        ["smart_publish", "pkg", "--dry-run", "--bump", "none"],
    )
    assert smart_publish.main() == 0


def test_main_success_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(
        smart_publish.sys,
        "argv",
        ["smart_publish", "pkg", "--dry-run", "--bump", "patch"],
    )
    assert smart_publish.main() == 0


def test_main_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(smart_publish, "run_publish_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        smart_publish.sys,
        "argv",
        [
            "smart_publish",
            "pkg",
            "--test",
            "--skip-existing",
            "--repository",
            "repo",
            "--repository-url",
            "http://example.com",
        ],
    )
    assert smart_publish.main() == 0


def test_main_update_version(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(
        smart_publish, "update_package_version", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(smart_publish, "run_publish_command", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        smart_publish.sys, "argv", ["smart_publish", "pkg", "--bump", "patch"]
    )
    assert smart_publish.main() == 0


def test_main_no_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(smart_publish, "build_distributions", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        smart_publish.sys, "argv", ["smart_publish", "pkg", "--no-upload"]
    )
    assert smart_publish.main() == 0


def test_main_no_upload_no_build(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(
        smart_publish.sys, "argv", ["smart_publish", "pkg", "--no-upload", "--no-build"]
    )
    assert smart_publish.main() == 0


def test_main_upload_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smart_publish, "find_package_root", lambda *args, **kwargs: tmp_path
    )
    (tmp_path / "setup.cfg").write_text("version = 1.2.3\n")
    monkeypatch.setattr(subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        smart_publish.sys, "argv", ["smart_publish", "pkg", "--no-build"]
    )
    assert smart_publish.main() == 0
