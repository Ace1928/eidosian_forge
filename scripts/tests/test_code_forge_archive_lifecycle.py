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


def test_run_archive_wave_can_retry_failed_batches(tmp_path: Path, monkeypatch) -> None:
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    (archive / "repo_a" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_a" / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    captured = {}

    def _fake_run_archive_ingestion_batches(**kwargs):
        captured.update(kwargs)
        return {"selected_batches": 1, "completed": 1, "failed": 0, "skipped": 0}

    monkeypatch.setattr(life_mod, "run_archive_ingestion_batches", _fake_run_archive_ingestion_batches)

    payload = life_mod.run_archive_wave(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        repo_keys=["repo_a"],
        batch_limit=1,
        progress_every=1,
        retry_failed=True,
    )

    assert payload["retry_failed"] is True
    assert captured["retry_failed"] is True
    assert captured["include_repo_keys"] == ["repo_a"]


def test_preview_retire_repo_can_assume_remove_mode(tmp_path: Path) -> None:
    plan_mod = _load_module("code_forge_archive_plan", "code_forge_archive_plan.py")
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_lifecycle"
    (archive / "repo_preview" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_preview" / "src" / "c.py").write_text("def c():\n    return 3\n", encoding="utf-8")

    plan_mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    db = CodeLibraryDB(repo / "data" / "code_forge" / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=repo / "data" / "code_forge" / "ingestion_runs")
    runner.ingest_path(archive / "repo_preview", extensions=[".py"], progress_every=1)

    batch_plan = json.loads((output_dir / "archive_ingestion_batches.json").read_text(encoding="utf-8"))
    state_path = output_dir / "archive_ingestion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for batch in batch_plan["batches"]:
        if batch.get("repo_key") == "repo_preview":
            batch_id = batch["batch_id"]
            state["batches"][batch_id]["status"] = "completed"
            batch_dir = output_dir / "batches" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "provenance_links.json").write_text("{}\n", encoding="utf-8")
            (batch_dir / "provenance_registry.json").write_text("{}\n", encoding="utf-8")
    state["completed_count"] = sum(1 for item in state["batches"].values() if item.get("status") == "completed")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    payload = life_mod.preview_retire_repos(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
        repo_keys=["repo_preview"],
        assume_remove_mode=True,
    )

    rec = payload["retirements"][0]
    assert rec["current_mode"] == "ingest_and_keep"
    assert rec["effective_mode"] == "ingest_and_remove"
    assert rec["retirement_ready"] is True
    assert rec["status"] == "ready"
    assert rec["parity_pass"] is True



def test_preview_retire_reports_missing_file_blockers(tmp_path: Path) -> None:
    plan_mod = _load_module("code_forge_archive_plan", "code_forge_archive_plan.py")
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_lifecycle"
    (archive / "repo_preview" / "src").mkdir(parents=True, exist_ok=True)
    (archive / "repo_preview" / "src" / "c.py").write_text("def c():\n    return 3\n", encoding="utf-8")
    (archive / "repo_preview" / "README.md").write_text("# docs\n", encoding="utf-8")

    plan_mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    db = CodeLibraryDB(repo / "data" / "code_forge" / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=repo / "data" / "code_forge" / "ingestion_runs")
    runner.ingest_path(archive / "repo_preview", extensions=[".py"], progress_every=1)

    batch_plan = json.loads((output_dir / "archive_ingestion_batches.json").read_text(encoding="utf-8"))
    state_path = output_dir / "archive_ingestion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for batch in batch_plan["batches"]:
        if batch.get("repo_key") == "repo_preview":
            batch_id = batch["batch_id"]
            state["batches"][batch_id]["status"] = "completed"
            batch_dir = output_dir / "batches" / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "provenance_links.json").write_text("{}\n", encoding="utf-8")
            (batch_dir / "provenance_registry.json").write_text("{}\n", encoding="utf-8")
    state["completed_count"] = sum(1 for item in state["batches"].values() if item.get("status") == "completed")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    payload = life_mod.preview_retire_repos(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
        repo_keys=["repo_preview"],
        assume_remove_mode=True,
    )

    rec = payload["retirements"][0]
    assert rec["status"] == "skipped"
    assert rec["retirement_ready"] is False
    assert rec["missing_file_count"] >= 1
    assert any(item.startswith("file_records 1/") for item in rec["blockers"])
    assert any(item.endswith("README.md") for item in rec["missing_file_samples"])


def test_preview_retire_reports_unindexed_source_files(tmp_path: Path) -> None:
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")
    repo_root = tmp_path / "forge"
    archive_root = repo_root / "archive_forge"
    target = archive_root / "sample_repo"
    target.mkdir(parents=True)
    (target / "kept.py").write_text("print(1)\n", encoding="utf-8")
    (target / "extra.ndjson").write_text('{"event":1}\n', encoding="utf-8")

    output_dir = repo_root / "data" / "code_forge" / "archive_ingestion" / "latest"
    output_dir.mkdir(parents=True)
    report_dir = repo_root / "reports"

    (output_dir / "repo_retention_policy.json").write_text(json.dumps({"default_mode": "ingest_and_remove", "repos": {}}), encoding="utf-8")
    (output_dir / "repo_index.json").write_text(json.dumps({"entries": [{"path": "sample_repo/kept.py", "repo_key": "sample_repo", "bytes": 9}]}), encoding="utf-8")
    (output_dir / "archive_ingestion_batches.json").write_text(json.dumps({"batches": [{"batch_id": "b1", "repo_key": "sample_repo", "route": "code_forge"}]}), encoding="utf-8")
    (output_dir / "archive_ingestion_state.json").write_text(json.dumps({"batches": {"b1": {"status": "completed", "route": "code_forge", "route_contract_version": "archive_code_forge_route_v1"}}}), encoding="utf-8")

    batch_dir = output_dir / "batches" / "b1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "provenance_links.json").write_text("{}", encoding="utf-8")
    (batch_dir / "provenance_registry.json").write_text("{}", encoding="utf-8")

    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    content_hash = db.add_text("print(1)\n")
    db.update_file_record(str((target / "kept.py").resolve()), content_hash, 3)

    report = life_mod.build_repo_status_report(
        repo_root=repo_root,
        archive_root=archive_root,
        output_dir=output_dir,
        report_dir=report_dir,
        repo_keys=["sample_repo"],
    )
    row = report["repos"][0]
    assert row["source_tree_file_count"] == 2
    assert row["source_tree_unindexed_count"] == 1
    assert "reversible_files 1/2" in row["retirement_blockers"]
    assert "unindexed_source_files 0/1" in row["retirement_blockers"]


def test_preview_retire_accepts_file_forge_coverage_for_unindexed_files(tmp_path: Path) -> None:
    life_mod = _load_module("code_forge_archive_lifecycle", "code_forge_archive_lifecycle.py")

    repo_root = tmp_path / "forge"
    archive_root = repo_root / "archive_forge"
    target = archive_root / "sample_repo"
    target.mkdir(parents=True)
    kept = target / "kept.py"
    extra = target / "extra.ndjson"
    kept.write_text("print(1)\n", encoding="utf-8")
    extra.write_text('{"event":1}\n', encoding="utf-8")

    output_dir = repo_root / "data" / "code_forge" / "archive_ingestion" / "latest"
    output_dir.mkdir(parents=True)
    report_dir = repo_root / "reports"

    (output_dir / "repo_retention_policy.json").write_text(json.dumps({"default_mode": "ingest_and_remove", "repos": {}}), encoding="utf-8")
    (output_dir / "repo_index.json").write_text(json.dumps({"entries": [{"path": "sample_repo/kept.py", "repo_key": "sample_repo", "bytes": 9}]}), encoding="utf-8")
    (output_dir / "archive_ingestion_batches.json").write_text(json.dumps({"batches": [{"batch_id": "b1", "repo_key": "sample_repo", "route": "code_forge"}]}), encoding="utf-8")
    (output_dir / "archive_ingestion_state.json").write_text(json.dumps({"batches": {"b1": {"status": "completed", "route": "code_forge", "route_contract_version": "archive_code_forge_route_v1"}}}), encoding="utf-8")

    batch_dir = output_dir / "batches" / "b1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "provenance_links.json").write_text("{}", encoding="utf-8")
    (batch_dir / "provenance_registry.json").write_text("{}", encoding="utf-8")

    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    content_hash = db.add_text("print(1)\n")
    db.update_file_record(str(kept.resolve()), content_hash, 3)

    from file_forge import FileForge
    forge = FileForge(base_path=repo_root)
    forge.index_directory(target, db_path=repo_root / "data" / "file_forge" / "library.sqlite")

    report = life_mod.build_repo_status_report(
        repo_root=repo_root,
        archive_root=archive_root,
        output_dir=output_dir,
        report_dir=report_dir,
        repo_keys=["sample_repo"],
    )
    row = report["repos"][0]
    assert row["reversible_file_count"] == 2
    assert row["source_tree_unindexed_reversible_count"] == 1
    assert row["retirement_ready"] is True
