from pathlib import Path

from code_forge.digester.pipeline import (
    build_duplication_index,
    build_repo_index,
    build_triage_report,
    run_archive_digester,
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
    assert payload.get("drift", {}).get("drift_report_json_path")
    validation = validate_output_dir(out)
    assert validation["pass"]
