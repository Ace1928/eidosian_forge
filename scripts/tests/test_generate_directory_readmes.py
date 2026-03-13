from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def test_generate_directory_readmes_writes_descendants(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(["init"], repo_root)
    _git(["config", "user.email", "test@example.com"], repo_root)
    _git(["config", "user.name", "Test"], repo_root)

    scribe = repo_root / "doc_forge" / "src" / "doc_forge" / "scribe"
    utils = repo_root / "doc_forge" / "src" / "doc_forge" / "utils"
    scribe.mkdir(parents=True, exist_ok=True)
    utils.mkdir(parents=True, exist_ok=True)
    (scribe / "service.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
    (utils / "paths.py").write_text("x = 1\n", encoding="utf-8")
    cfg_dir = repo_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["doc_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], repo_root)
    _git(["commit", "-m", "init"], repo_root)

    summary_path = repo_root / "summary.json"
    script = Path("/data/data/com.termux/files/home/eidosian_forge/scripts/generate_directory_readmes.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--path",
            "doc_forge/src/doc_forge",
            "--missing-only",
            "--summary-json",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (scribe / "README.md").exists()
    assert (utils / "README.md").exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    written = {row["path"] for row in payload["writes"]}
    assert "doc_forge/src/doc_forge/scribe" in written
    assert "doc_forge/src/doc_forge/utils" in written


def test_generate_directory_readmes_managed_only_does_not_clobber_manual_readmes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(["init"], repo_root)
    _git(["config", "user.email", "test@example.com"], repo_root)
    _git(["config", "user.name", "Test"], repo_root)

    package = repo_root / "doc_forge" / "src" / "doc_forge"
    package.mkdir(parents=True, exist_ok=True)
    (package / "README.md").write_text("# Manual README\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    cfg_dir = repo_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["doc_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], repo_root)
    _git(["commit", "-m", "init"], repo_root)

    summary_path = repo_root / "summary.json"
    script = Path("/data/data/com.termux/files/home/eidosian_forge/scripts/generate_directory_readmes.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--path",
            "doc_forge/src/doc_forge",
            "--managed-only",
            "--summary-json",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (package / "README.md").read_text(encoding="utf-8") == "# Manual README\n"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["write_count"] == 0
