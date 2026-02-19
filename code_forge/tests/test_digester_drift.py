import json
from pathlib import Path

from code_forge.digester.pipeline import run_archive_digester
from code_forge.digester.schema import validate_output_dir
from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def _make_repo(root: Path, *, include_extra: bool = False) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "m.py").write_text(
        "def score(values):\n"
        "    return sum(values)\n",
        encoding="utf-8",
    )
    if include_extra:
        (root / "src" / "n.py").write_text(
            "def score(values):\n"
            "    total = sum(values)\n"
            "    return total\n",
            encoding="utf-8",
        )


def test_run_archive_digester_writes_drift_and_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo, include_extra=False)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    out = tmp_path / "digester"

    first = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
    )
    assert (out / "drift_report.json").exists()
    assert first.get("drift", {}).get("history_snapshot_path")

    _make_repo(repo, include_extra=True)
    second = run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
    )
    comparison = second.get("drift", {}).get("comparison", {})
    assert comparison.get("compared_metric_count", 0) > 0
    assert isinstance(comparison.get("comparisons"), list)


def test_validate_output_dir_fails_for_invalid_optional_drift_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo, include_extra=False)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    out = tmp_path / "digester"

    run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
    )

    drift_path = out / "drift_report.json"
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    drift.pop("comparison", None)
    drift_path.write_text(json.dumps(drift, indent=2) + "\n", encoding="utf-8")

    report = validate_output_dir(out)
    assert not report["pass"]
    assert any("drift_report.json" in err for err in report["errors"])
