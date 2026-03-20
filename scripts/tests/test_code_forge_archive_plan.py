from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module() -> object:
    path = Path(__file__).resolve().parents[1] / "code_forge_archive_plan.py"
    spec = importlib.util.spec_from_file_location("code_forge_archive_plan", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_archive_plan_report_builds_full_archive_plan(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    report_dir = repo / "reports" / "code_forge_archive_plan"
    (archive / "src").mkdir(parents=True, exist_ok=True)
    (archive / "docs").mkdir(parents=True, exist_ok=True)
    (archive / "meta").mkdir(parents=True, exist_ok=True)
    (repo / "data" / "code_forge" / "vectors").mkdir(parents=True, exist_ok=True)
    (repo / ".gitattributes").write_text(
        "data/code_forge/**/*.bin filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (archive / "src" / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (archive / "docs" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (archive / "meta" / "config.json").write_text('{"ok": true}\n', encoding="utf-8")
    (repo / "data" / "code_forge" / "vectors" / "index.bin").write_bytes(b"1234567890")

    result = mod.write_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        report_dir=report_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    report = result["report"]
    assert report["archive_files_total"] == 3
    assert report["batch_count"] >= 3
    assert report["route_batch_counts"]["code_forge"] >= 1
    assert report["route_batch_counts"]["document_pipeline"] >= 1
    assert report["route_batch_counts"]["knowledge_metadata"] >= 1
    assert report["repo_count"] == 3
    assert report["retirement"]["ready"] is False
    assert report["provenance"]["provenance_link_count"] == 0
    assert report["git_lfs_patterns"]
    assert report["vector_graph_assets"][0]["path"] == "data/code_forge/vectors/index.bin"
    assert Path(result["latest_json"]).exists()
    assert Path(result["latest_markdown"]).exists()
    assert (output_dir / "archive_ingestion_state.json").exists()


def test_build_archive_plan_report_preserves_existing_batch_progress_on_refresh(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path / "repo"
    archive = repo / "archive_forge"
    output_dir = repo / "data" / "code_forge" / "archive_ingestion" / "latest"
    (archive / "src").mkdir(parents=True, exist_ok=True)
    (archive / "src" / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (archive / "src" / "beta.py").write_text("def beta():\n    return 2\n", encoding="utf-8")

    first = mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )
    state_path = output_dir / "archive_ingestion_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    first_batch = next(iter(state["batches"]))
    state["batches"][first_batch]["status"] = "completed"
    state["batches"][first_batch]["attempts"] = 2
    state["completed_count"] = 1
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    second = mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    assert first["batch_count"] == second["batch_count"]
    assert second["completed_count"] == 1
    assert second["state_status_counts"]["completed"] == 1

    (archive / "src" / "beta.py").write_text("def beta():\n    return 3\n", encoding="utf-8")
    third = mod.build_archive_plan_report(
        repo_root=repo,
        archive_root=archive,
        output_dir=output_dir,
        refresh=True,
        max_files_per_batch=1,
        max_bytes_per_batch=4096,
    )

    assert third["change_summary"]["modified_count"] == 1
    assert third["state_status_counts"]["pending"] >= 1
