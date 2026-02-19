from pathlib import Path

from code_forge.bench.runner import run_benchmark_suite


def _make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text(
        "import math\n"
        "def calc(v):\n"
        "    return math.floor(v)\n",
        encoding="utf-8",
    )


def test_run_benchmark_suite_and_baseline(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db_path = tmp_path / "library.sqlite"
    runs_dir = tmp_path / "runs"
    out = tmp_path / "bench.json"
    baseline = tmp_path / "baseline.json"

    first = run_benchmark_suite(
        root_path=repo,
        db_path=db_path,
        runs_dir=runs_dir,
        output_path=out,
        extensions=[".py"],
        max_files=10,
        ingestion_repeats=1,
        query_repeats=2,
        queries=["calc", "math floor"],
        baseline_path=baseline,
        max_regression_pct=50.0,
        prepare_ingest=True,
        write_baseline=True,
    )

    assert out.exists()
    assert baseline.exists()
    assert first["gate"]["pass"]

    second = run_benchmark_suite(
        root_path=repo,
        db_path=db_path,
        runs_dir=runs_dir,
        output_path=out,
        extensions=[".py"],
        max_files=10,
        ingestion_repeats=1,
        query_repeats=2,
        queries=["calc", "math floor"],
        baseline_path=baseline,
        max_regression_pct=99.0,
        prepare_ingest=True,
        write_baseline=False,
    )
    assert second["gate"]["baseline_loaded"]
