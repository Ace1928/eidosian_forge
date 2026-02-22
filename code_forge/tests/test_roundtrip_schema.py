from __future__ import annotations

import json
from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB
from code_forge.reconstruct.pipeline import run_roundtrip_pipeline
from code_forge.reconstruct.schema import validate_roundtrip_workspace


def _make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text(
        "def f(v):\n" "    return v + 1\n",
        encoding="utf-8",
    )


def test_validate_roundtrip_workspace_pass(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    workspace = tmp_path / "roundtrip"

    run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        strict_validation=True,
        apply=False,
    )

    report = validate_roundtrip_workspace(workspace, verify_hashes=True)
    assert report["pass"] is True
    assert "provenance_links.json" in report["files"]
    assert "provenance_registry.json" in report["files"]


def test_validate_roundtrip_workspace_fail_on_mutated_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    workspace = tmp_path / "roundtrip"

    run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        strict_validation=True,
        apply=False,
    )

    manifest_path = workspace / "reconstructed" / "reconstruction_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"][0]["written_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = validate_roundtrip_workspace(workspace, verify_hashes=True)
    assert report["pass"] is False
    assert any("hash mismatch" in err for err in report["errors"])


def test_validate_roundtrip_workspace_fail_on_missing_signature(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    workspace = tmp_path / "roundtrip"

    run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        strict_validation=True,
        apply=False,
    )

    summary_path = workspace / "roundtrip_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("signature", None)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = validate_roundtrip_workspace(workspace, verify_hashes=False)
    assert report["pass"] is False
    assert any("summary.signature must be object" in err for err in report["errors"])
