from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def _load_module(name: str, rel_name: str) -> object:
    path = Path(__file__).resolve().parents[1] / rel_name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_status_report_and_retirement_readiness(tmp_path: Path) -> None:
    plan_mod = _load_module("code_forge_archive_plan", "code_forge_archive_plan.py")
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_lifecycle"
    (archive / "repo_keep" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_drop" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_keep" / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (archive / "repo_drop" / "src" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    plan_mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )
    life_mod.set_repo_mode(output_dir / "repo_retention_policy.json", "repo_drop", "ingest_and_remove", "cleanup")

    db = CodeLibraryDB(repo / "data" / "code_forge" / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=repo / "data" / "code_forge" / "ingestion_runs")
    runner.ingest_path(archive / "repo_drop", extensions=[".py"], progress_every=1)

    batch_plan = json.loads((output_dir / "archive_ingestion_batches.json").read_text(encoding="utf-8"))
    state_path = output_dir / "archive_ingestion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for batch in batch_plan["batches"]:
        if batch.get("repo_key") == "repo_drop":
            batch_id = batch["batch_id"]
            state["batches"][batch_id]["status"] = "completed"
            batch_dir = output_dir / "batches" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "provenance_links.json").write_text("{}\n", encoding="utf-8")
            (batch_dir / "provenance_registry.json").write_text("{}\n", encoding="utf-8")
    state["completed_count"] = sum(1 for item in state["batches"].values() if item.get("status") == "completed")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    report = life_mod.build_repo_status_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
    )
    rows = {row["repo_key"]: row for row in report["repos"]}
    assert rows["repo_drop"]["mode"] == "ingest_and_remove"
    assert rows["repo_drop"]["retirement_ready"] is True
    assert rows["repo_keep"]["mode"] == "ingest_and_keep"
    assert rows["repo_keep"]["retirement_ready"] is False


def test_retire_repo_moves_source_reversibly(tmp_path: Path) -> None:
    plan_mod = _load_module("code_forge_archive_plan", "code_forge_archive_plan.py")
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_lifecycle"
    (archive / "repo_drop" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_drop" / "src" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    plan_mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )
    life_mod.set_repo_mode(output_dir / "repo_retention_policy.json", "repo_drop", "ingest_and_remove", "cleanup")

    db = CodeLibraryDB(repo / "data" / "code_forge" / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=repo / "data" / "code_forge" / "ingestion_runs")
    runner.ingest_path(archive / "repo_drop", extensions=[".py"], progress_every=1)

    batch_plan = json.loads((output_dir / "archive_ingestion_batches.json").read_text(encoding="utf-8"))
    state_path = output_dir / "archive_ingestion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for batch in batch_plan["batches"]:
        if batch.get("repo_key") == "repo_drop":
            batch_id = batch["batch_id"]
            state["batches"][batch_id]["status"] = "completed"
            batch_dir = output_dir / "batches" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "provenance_links.json").write_text("{}\n", encoding="utf-8")
            (batch_dir / "provenance_registry.json").write_text("{}\n", encoding="utf-8")
    state["completed_count"] = sum(1 for item in state["batches"].values() if item.get("status") == "completed")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    payload = life_mod.retire_repos(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
        repo_keys=["repo_drop"],
        dry_run=False,
    )
    rec = payload["retirements"][0]
    assert rec["status"] == "retired"
    assert not (archive / "repo_drop").exists()
    assert Path(rec["retired_root"]).exists()
    assert Path(rec["reconstruction_manifest_path"]).exists()
    assert Path(rec["parity_report_path"]).exists()


def test_ensure_archive_plan_reuses_existing_report(tmp_path: Path) -> None:
    plan_mod = _load_module("code_forge_archive_plan", "code_forge_archive_plan.py")
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_plan"
    (archive / "repo_keep" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_keep" / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    report = plan_mod.write_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    original = life_mod.build_archive_plan_report
    def _boom(*args, **kwargs):
        raise AssertionError("unexpected refresh")
    life_mod.build_archive_plan_report = _boom
    try:
        payload = life_mod._ensure_archive_plan(
            repo_root=repo,
            archive_root=archive,
            output_dir=output_dir,
            refresh=False,
        )
    finally:
        life_mod.build_archive_plan_report = original

    assert payload["archive_files_total"] == report["report"]["archive_files_total"]
    assert payload["batch_count"] == report["report"]["batch_count"]
