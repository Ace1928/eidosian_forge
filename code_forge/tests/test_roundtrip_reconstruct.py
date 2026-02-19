from __future__ import annotations

import shutil
from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB
from code_forge.reconstruct.pipeline import (
    apply_reconstruction,
    build_reconstruction_from_library,
    compare_tree_parity,
    run_roundtrip_pipeline,
)


def _make_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text('{"name":"mini","version":1}\n', encoding="utf-8")


def test_file_record_prefix_and_reconstruction_parity(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    _make_repo(repo_a)
    _make_repo(repo_b)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    runner.ingest_path(repo_a, extensions=[".py", ".json"], progress_every=1)
    runner.ingest_path(repo_b, extensions=[".py", ".json"], progress_every=1)

    records_a = list(db.iter_file_records(path_prefix=str(repo_a)))
    assert records_a
    assert all(str(rec["file_path"]).startswith(str(repo_a)) for rec in records_a)
    assert db.count_file_records(path_prefix=str(repo_a)) == len(records_a)

    reconstructed = tmp_path / "reconstructed_a"
    manifest = build_reconstruction_from_library(
        db=db,
        source_root=repo_a,
        output_dir=reconstructed,
        strict=True,
    )
    assert manifest["files_written"] == len(records_a)
    assert Path(manifest["manifest_path"]).exists()

    parity = compare_tree_parity(
        source_root=repo_a,
        reconstructed_root=reconstructed,
        report_path=tmp_path / "parity.json",
    )
    assert parity["pass"] is True
    assert Path(parity["report_path"]).exists()


def test_apply_reconstruction_transactional_backup(tmp_path: Path) -> None:
    target = tmp_path / "target_repo"
    target.mkdir()
    _make_repo(target)

    reconstructed = tmp_path / "reconstructed_repo"
    shutil.copytree(target, reconstructed)
    (reconstructed / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b + 1\n",
        encoding="utf-8",
    )
    (reconstructed / "config.json").unlink()
    (reconstructed / "new.txt").write_text("new-file\n", encoding="utf-8")

    backup_root = tmp_path / "backups"
    report = apply_reconstruction(
        reconstructed_root=reconstructed,
        target_root=target,
        backup_root=backup_root,
        parity_report={"pass": False},
        require_parity_pass=False,
        prune=True,
    )
    assert report["changed_or_new_count"] == 2
    assert report["removed_count"] == 1
    assert report["backup_count"] == 2
    tx_dir = Path(report["backup_transaction_dir"])
    assert tx_dir.exists()
    assert (tx_dir / "apply_report.json").exists()
    assert (target / "pkg" / "core.py").read_text(encoding="utf-8").endswith("return a + b + 1\n")
    assert not (target / "config.json").exists()
    assert (target / "new.txt").exists()

    noop = apply_reconstruction(
        reconstructed_root=reconstructed,
        target_root=target,
        backup_root=backup_root,
        parity_report={"pass": True},
        require_parity_pass=True,
        prune=True,
    )
    assert noop["noop"] is True


def test_roundtrip_pipeline_end_to_end(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    workspace = tmp_path / "roundtrip"

    summary = run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace,
        mode="analysis",
        extensions=[".py", ".json"],
        progress_every=1,
        strict_validation=True,
        apply=False,
    )
    assert summary["parity_pass"] is True
    assert Path(summary["summary_path"]).exists()
    assert (workspace / "digester" / "archive_digester_summary.json").exists()
    assert (workspace / "reconstructed" / "reconstruction_manifest.json").exists()
    assert (workspace / "parity_report.json").exists()


def test_roundtrip_uses_latest_effective_run_when_no_new_units(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)
    kb = tmp_path / "kb.json"

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    workspace_1 = tmp_path / "roundtrip_1"
    workspace_2 = tmp_path / "roundtrip_2"

    first = run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace_1,
        mode="analysis",
        extensions=[".py", ".json"],
        progress_every=1,
        strict_validation=True,
        apply=False,
        sync_knowledge_path=kb,
        graphrag_output_dir=workspace_1 / "graphrag",
        graph_export_limit=200,
    )
    first_run_id = str((first["digest"]["ingestion_stats"] or {}).get("run_id"))
    assert first["digest"]["knowledge_sync"]["scanned_units"] > 0
    assert "exported" in first["digest"]["graphrag_export"]

    second = run_roundtrip_pipeline(
        root_path=repo,
        db=db,
        runner=runner,
        workspace_dir=workspace_2,
        mode="analysis",
        extensions=[".py", ".json"],
        progress_every=1,
        strict_validation=True,
        apply=False,
        sync_knowledge_path=kb,
        graphrag_output_dir=workspace_2 / "graphrag",
        graph_export_limit=200,
    )
    second_stats = second["digest"]["ingestion_stats"]
    assert second_stats["units_created"] == 0
    assert second["digest"]["integration_run_id"] == first_run_id
    assert second["digest"]["knowledge_sync"]["scanned_units"] > 0
    assert "exported" in second["digest"]["graphrag_export"]
