import json
from pathlib import Path

from code_forge.digester import pipeline as digester_pipeline
from code_forge.digester.pipeline import (
    build_archive_ingestion_batches,
    build_archive_reduction_plan,
    build_duplication_index,
    build_repo_index,
    build_triage_dashboard,
    build_triage_report,
    initialize_archive_ingestion_state,
    load_archive_ingestion_state,
    run_archive_digester,
    run_archive_ingestion_batches,
    update_archive_ingestion_state,
)
from code_forge.digester.schema import validate_output_dir
from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def _make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)

    shared = "def util(items):\n    total = sum(items)\n    return total\n"
    (root / "src" / "a.py").write_text(shared, encoding="utf-8")
    (root / "src" / "b.py").write_text(shared, encoding="utf-8")
    (root / "tests" / "test_a.py").write_text(
        "from src.a import util\n\ndef test_util():\n    assert util([1,2]) == 3\n",
        encoding="utf-8",
    )


def test_build_repo_and_duplication_and_triage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    runner.ingest_path(repo, extensions=[".py"], progress_every=1)

    output = tmp_path / "digester"
    repo_index = build_repo_index(repo, output, extensions=[".py"])
    duplication = build_duplication_index(db, output, near_min_tokens=1, near_limit=50)
    triage = build_triage_report(db, repo_index, duplication, output)

    assert (output / "repo_index.json").exists()
    assert (output / "duplication_index.json").exists()
    assert (output / "triage.json").exists()
    assert (output / "triage_audit.json").exists()
    assert (output / "triage.csv").exists()
    assert (output / "triage_report.md").exists()
    assert duplication["summary"]["structural_group_count"] >= 1

    assert repo_index["files_total"] == 3
    assert triage["entries"]
    assert all("confidence" in rec and "rule_id" in rec for rec in triage["entries"])
    labels = {rec["label"] for rec in triage["entries"]}
    assert labels.intersection({"extract", "delete_candidate", "keep", "refactor", "quarantine"})


def test_run_archive_digester_end_to_end(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    out = tmp_path / "output"

    payload = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        max_files=None,
        progress_every=1,
    )

    assert payload["ingestion_stats"]["files_processed"] >= 1
    assert (out / "archive_digester_summary.json").exists()
    assert (out / "repo_index.json").exists()
    assert (out / "duplication_index.json").exists()
    assert (out / "dependency_graph.json").exists()
    assert (out / "triage.json").exists()
    assert (out / "drift_report.json").exists()
    assert (out / "provenance_links.json").exists()
    assert (out / "provenance_registry.json").exists()
    assert payload.get("provenance_path")
    assert payload.get("provenance_registry_path")
    assert payload.get("drift", {}).get("drift_report_json_path")
    validation = validate_output_dir(out)
    assert validation["pass"]

    reduction = build_archive_reduction_plan(out, max_delete_candidates=10, max_extract_candidates=10)
    assert reduction["counts"]["entries_total"] >= 1
    assert (out / "archive_reduction_plan.json").exists()
    assert (out / "archive_reduction_plan.md").exists()
    dashboard = build_triage_dashboard(out, max_rows=20)
    assert dashboard["entries_total"] >= 1
    dash_path = out / "dashboard.html"
    assert dash_path.exists()
    assert "Code Forge Dashboard" in dash_path.read_text(encoding="utf-8")


def test_run_archive_digester_integration_policy_modes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)
    kb = tmp_path / "kb.json"
    memory = tmp_path / "episodic_memory.json"

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")

    first = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=tmp_path / "out_first",
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        sync_knowledge_path=kb,
        sync_memory_path=memory,
        graphrag_output_dir=tmp_path / "grag_first",
        graph_export_limit=200,
        integration_policy="effective_run",
    )
    first_run_id = str((first["ingestion_stats"] or {}).get("run_id"))

    run_policy = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=tmp_path / "out_run",
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        sync_knowledge_path=kb,
        sync_memory_path=memory,
        graphrag_output_dir=tmp_path / "grag_run",
        graph_export_limit=200,
        integration_policy="run",
    )
    assert run_policy["integration_policy"] == "run"
    assert run_policy["integration_run_id"] == str((run_policy["ingestion_stats"] or {}).get("run_id"))

    effective_policy = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=tmp_path / "out_effective",
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        sync_knowledge_path=kb,
        sync_memory_path=memory,
        graphrag_output_dir=tmp_path / "grag_effective",
        graph_export_limit=200,
        integration_policy="effective_run",
    )
    assert (effective_policy["ingestion_stats"] or {}).get("units_created") == 0
    assert effective_policy["integration_policy"] == "effective_run"
    assert effective_policy["integration_run_id"] == first_run_id

    global_policy = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=tmp_path / "out_global",
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        sync_knowledge_path=kb,
        sync_memory_path=memory,
        graphrag_output_dir=tmp_path / "grag_global",
        graph_export_limit=200,
        integration_policy="global",
    )
    assert global_policy["integration_policy"] == "global"
    assert global_policy["integration_run_id"] is None
    provenance_path = tmp_path / "out_global" / "provenance_links.json"
    registry_path = tmp_path / "out_global" / "provenance_registry.json"
    assert provenance_path.exists()
    assert registry_path.exists()
    assert (global_policy.get("memory_sync") or {}).get("scanned_units", 0) > 0
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert isinstance(provenance.get("memory_links"), dict)
    assert registry.get("schema_version")
    unit_links = ((registry.get("links") or {}).get("unit_links")) or []
    assert isinstance(unit_links, list)


def test_build_archive_batches_and_state(tmp_path: Path) -> None:
    repo = tmp_path / "archive_like"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "images").mkdir()
    (repo / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (repo / "docs" / "meta.json").write_text('{"ok": true}\n', encoding="utf-8")
    (repo / "images" / "img.webp").write_bytes(b"RIFF0000WEBP")

    output = tmp_path / "digester"
    repo_index = build_repo_index(repo, output, extensions=[".py", ".md", ".json", ".webp"])
    batches = build_archive_ingestion_batches(repo_index, output, max_files_per_batch=1, max_bytes_per_batch=1024)
    state = initialize_archive_ingestion_state(batches, output)

    assert (output / "archive_ingestion_batches.json").exists()
    assert (output / "archive_ingestion_state.json").exists()
    assert batches["batch_count"] >= 4
    assert "code_forge" in batches["route_counts"]
    assert "document_pipeline" in batches["route_counts"]
    assert "knowledge_metadata" in batches["route_counts"]
    assert "defer_binary" in batches["route_counts"]

    first_batch_id = batches["batches"][0]["batch_id"]
    updated = update_archive_ingestion_state(output, batch_id=first_batch_id, status="completed")
    assert updated["completed_count"] == 1
    assert state["batch_count"] == batches["batch_count"]
    loaded = load_archive_ingestion_state(output)
    assert loaded["completed_count"] == 1


def test_build_repo_index_include_paths_filters_entries(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "drop.py").write_text("def drop():\n    return 0\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")

    output = tmp_path / "digester"
    repo_index = build_repo_index(
        repo,
        output,
        extensions=[".py", ".md"],
        include_paths=["src/keep.py", "docs/guide.md"],
    )

    assert repo_index["files_total"] == 2
    assert {entry["path"] for entry in repo_index["entries"]} == {"src/keep.py", "docs/guide.md"}


def test_run_archive_ingestion_batches_processes_code_doc_and_metadata_routes(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "archive_like"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "meta").mkdir()
    (repo / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("# Guide\n\nAlpha beta gamma delta.\n", encoding="utf-8")
    (repo / "meta" / "index.json").write_text('{"name": "guide", "status": "ready"}\n', encoding="utf-8")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    output = tmp_path / "digester"
    kb = tmp_path / "kb.json"
    grag = tmp_path / "grag"

    repo_index = build_repo_index(repo, output, extensions=[".py", ".md", ".json"])
    batch_plan = build_archive_ingestion_batches(repo_index, output, max_files_per_batch=1, max_bytes_per_batch=4096)
    initialize_archive_ingestion_state(batch_plan, output)

    monkeypatch.setattr(
        digester_pipeline,
        "_process_document_batch",
        lambda **kwargs: {
            "generated_at": "now",
            "batch_id": kwargs["batch"]["batch_id"],
            "route": "document_pipeline",
            "files_processed": len(kwargs["batch"]["paths"]),
            "nodes_created": 2,
            "lexicon_nodes_added": 3,
            "results": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        digester_pipeline,
        "_process_metadata_batch",
        lambda **kwargs: {
            "generated_at": "now",
            "batch_id": kwargs["batch"]["batch_id"],
            "route": "knowledge_metadata",
            "files_processed": len(kwargs["batch"]["paths"]),
            "nodes_created": 1,
            "lexicon_nodes_added": 1,
            "results": [],
            "errors": [],
        },
    )

    summary = run_archive_ingestion_batches(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=output,
        kb_path=kb,
        graphrag_output_dir=grag,
        include_routes=["code_forge", "document_pipeline", "knowledge_metadata"],
    )

    assert summary["completed"] >= 3
    assert (output / "archive_ingestion_wave_summary.json").exists()
    batches_dir = output / "batches"
    assert any(path.name == "archive_digester_summary.json" for path in batches_dir.rglob("archive_digester_summary.json"))
    assert len(list(batches_dir.rglob("batch_summary.json"))) >= 3
    routes = {str(run.get("route") or "") for run in summary["runs"]}
    assert {"code_forge", "document_pipeline", "knowledge_metadata"}.issubset(routes)


def test_run_archive_ingestion_batches_filters_repo_keys(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "archive_like"
    archive.mkdir()
    (archive / "repo_a" / "src").mkdir(parents=True)
    (archive / "repo_b" / "src").mkdir(parents=True)
    (archive / "repo_a" / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (archive / "repo_b" / "src" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    output = tmp_path / "digester"
    kb = tmp_path / "kb.json"

    plan = {
        "generated_at": "now",
        "root_path": str(archive),
        "files_total": 2,
        "batch_count": 2,
        "route_counts": {"code_forge": 2},
        "batches": [
            {
                "batch_id": "batch_a",
                "repo_key": "repo_a",
                "route": "code_forge",
                "category": "source",
                "sequence": 1,
                "file_count": 1,
                "total_bytes": 16,
                "extensions": [".py"],
                "paths": ["repo_a/src/a.py"],
                "status": "pending",
            },
            {
                "batch_id": "batch_b",
                "repo_key": "repo_b",
                "route": "code_forge",
                "category": "source",
                "sequence": 1,
                "file_count": 1,
                "total_bytes": 16,
                "extensions": [".py"],
                "paths": ["repo_b/src/b.py"],
                "status": "pending",
            },
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "archive_ingestion_batches.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    initialize_archive_ingestion_state(plan, output)

    calls: list[list[str]] = []

    def _fake_run_archive_digester(**kwargs):
        calls.append(list(kwargs.get("include_paths") or []))
        return {
            "generated_at": "now",
            "route": "code_forge",
            "ingestion_stats": {"files_processed": len(kwargs.get("include_paths") or [])},
        }

    monkeypatch.setattr(digester_pipeline, "run_archive_digester", _fake_run_archive_digester)

    summary = run_archive_ingestion_batches(
        root_path=archive,
        db=db,
        runner=runner,
        output_dir=output,
        kb_path=kb,
        include_routes=["code_forge"],
        include_repo_keys=["repo_b"],
    )

    assert summary["selected_batches"] == 1
    assert calls == [["repo_b/src/b.py"]]
    state = load_archive_ingestion_state(output)
    assert state["batches"]["batch_a"]["status"] == "pending"
    assert state["batches"]["batch_b"]["status"] == "completed"


def test_triage_profile_hot_path_preserves_duplicate_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    runner.ingest_path(repo, extensions=[".py"], progress_every=1)

    output = tmp_path / "digester"
    repo_index = build_repo_index(repo, output, extensions=[".py"])
    duplication = build_duplication_index(db, output, near_min_tokens=1, near_limit=50)
    triage_without_profile = build_triage_report(db, repo_index, duplication, output)

    profile = {
        "file_hotness": {
            "src/a.py": 1.4,
        }
    }
    profile_path = tmp_path / "profile_hotspots.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    triage_with_profile = build_triage_report(
        db,
        repo_index,
        duplication,
        output,
        profile_hotspots={"src/a.py": {"hotness": 1.4, "samples": 2, "source": str(profile_path)}},
    )

    by_file_no_profile = {rec["file_path"]: rec for rec in triage_without_profile["entries"]}
    by_file_profile = {rec["file_path"]: rec for rec in triage_with_profile["entries"]}
    assert "src/a.py" in by_file_profile
    assert by_file_profile["src/a.py"]["metrics"]["profile_hotness"] > 0.0
    assert by_file_profile["src/a.py"]["rule_id"].startswith("RULE_H")
    if by_file_no_profile.get("src/a.py", {}).get("label") == "delete_candidate":
        assert by_file_profile["src/a.py"]["label"] != "delete_candidate"


def test_run_archive_ingestion_batches_requeues_completed_batch_when_files_change(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "archive_like"
    archive.mkdir()
    (archive / "repo_a" / "src").mkdir(parents=True)
    source = archive / "repo_a" / "src" / "a.py"
    source.write_text("def a():\n    return 1\n", encoding="utf-8")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    output = tmp_path / "digester"
    kb = tmp_path / "kb.json"

    plan = {
        "generated_at": "now",
        "root_path": str(archive),
        "files_total": 1,
        "batch_count": 1,
        "route_counts": {"code_forge": 1},
        "batches": [
            {
                "batch_id": "batch_a",
                "repo_key": "repo_a",
                "route": "code_forge",
                "category": "source",
                "sequence": 1,
                "file_count": 1,
                "total_bytes": 16,
                "extensions": [".py"],
                "paths": ["repo_a/src/a.py"],
                "status": "pending",
            }
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "archive_ingestion_batches.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    initialize_archive_ingestion_state(plan, output)

    calls: list[list[str]] = []

    def _fake_run_archive_digester(**kwargs):
        calls.append(list(kwargs.get("include_paths") or []))
        return {
            "generated_at": "now",
            "route": "code_forge",
            "ingestion_stats": {"files_processed": len(kwargs.get("include_paths") or [])},
        }

    monkeypatch.setattr(digester_pipeline, "run_archive_digester", _fake_run_archive_digester)

    first = run_archive_ingestion_batches(
        root_path=archive,
        db=db,
        runner=runner,
        output_dir=output,
        kb_path=kb,
        include_routes=["code_forge"],
        include_repo_keys=["repo_a"],
    )
    assert first["completed"] == 1

    source.write_text("def a():\n    return 2\n", encoding="utf-8")

    second = run_archive_ingestion_batches(
        root_path=archive,
        db=db,
        runner=runner,
        output_dir=output,
        kb_path=kb,
        include_routes=["code_forge"],
        include_repo_keys=["repo_a"],
    )

    assert second["selected_batches"] == 1
    assert calls == [["repo_a/src/a.py"], ["repo_a/src/a.py"]]
    state = load_archive_ingestion_state(output)
    assert state["batches"]["batch_a"]["status"] == "completed"
    assert state["batches"]["batch_a"]["content_signature"]


def test_run_archive_ingestion_batches_writes_batch_error_artifact(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "archive_like"
    archive.mkdir()
    (archive / "repo_a" / "src").mkdir(parents=True)
    (archive / "repo_a" / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    output = tmp_path / "digester"
    kb = tmp_path / "kb.json"

    plan = {
        "generated_at": "now",
        "root_path": str(archive),
        "files_total": 1,
        "batch_count": 1,
        "route_counts": {"code_forge": 1},
        "batches": [
            {
                "batch_id": "batch_a",
                "repo_key": "repo_a",
                "route": "code_forge",
                "category": "source",
                "sequence": 1,
                "file_count": 1,
                "total_bytes": 16,
                "extensions": [".py"],
                "paths": ["repo_a/src/a.py"],
                "status": "pending",
            }
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "archive_ingestion_batches.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    initialize_archive_ingestion_state(plan, output)

    def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(digester_pipeline, "run_archive_digester", _boom)

    summary = run_archive_ingestion_batches(
        root_path=archive,
        db=db,
        runner=runner,
        output_dir=output,
        kb_path=kb,
        include_routes=["code_forge"],
        include_repo_keys=["repo_a"],
    )

    assert summary["failed"] == 1
    error_path = output / "batches" / "batch_a" / "batch_error.json"
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert payload["error"] == "boom"
    assert "RuntimeError: boom" in payload["traceback"]
    state = load_archive_ingestion_state(output)
    assert state["batches"]["batch_a"]["status"] == "failed"
    assert state["batches"]["batch_a"]["summary_path"].endswith("batch_error.json")


def test_run_archive_digester_degrades_when_knowledge_sync_unavailable(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    out = tmp_path / "output"
    kb = tmp_path / "kb.json"

    monkeypatch.setattr(digester_pipeline, "sync_units_to_knowledge_forge", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("embedding offline")))

    payload = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
        sync_knowledge_path=kb,
    )

    assert payload["ingestion_stats"]["files_processed"] >= 1
    assert payload["knowledge_sync"]["status"] == "error"
    assert payload["integration_errors"][0]["stage"] == "knowledge_sync"


def test_process_metadata_batch_degrades_when_knowledge_forge_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "archive_like"
    archive.mkdir()
    (archive / "meta").mkdir()
    (archive / "meta" / "index.json").write_text('{"name": "guide", "status": "ready"}\n', encoding="utf-8")
    output = tmp_path / "digester"
    kb = tmp_path / "kb.json"

    from knowledge_forge.core.graph import KnowledgeForge

    monkeypatch.setattr(KnowledgeForge, "add_knowledge", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("embedding offline")))
    monkeypatch.setattr(digester_pipeline, "_word_forge_seed_from_text", lambda text: {"status": "success", "nodes_added": 1, "edges_added": 0})

    payload = digester_pipeline._process_metadata_batch(
        root_path=archive,
        batch={"batch_id": "batch_meta", "route": "knowledge_metadata", "paths": ["meta/index.json"]},
        output_dir=output,
        kb_path=kb,
    )

    assert payload["files_processed"] == 0
    assert payload["errors"][0]["stage"] == "knowledge_forge"
    assert payload["results"][0]["node_id"] is None
    assert payload["results"][0]["node_error"] == "embedding offline"
