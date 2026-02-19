from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def test_ingest_runner_manifest(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    file_path = root / "mod.py"
    file_path.write_text("def hi():\n    return 'hi'\n")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runs_dir = tmp_path / "runs"
    runner = IngestionRunner(db=db, runs_dir=runs_dir)

    stats = runner.ingest_path(root, mode="analysis")

    manifest = runs_dir / f"{stats.run_id}.json"
    assert manifest.exists()
    data = manifest.read_text()
    assert "mod.py" in data
    assert stats.files_processed == 1
    assert stats.units_created >= 2


def test_ingest_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    file_path = root / "mod.py"
    file_path.write_text("def hi():\n    return 'hi'\n")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runs_dir = tmp_path / "runs"
    runner = IngestionRunner(db=db, runs_dir=runs_dir)

    stats_first = runner.ingest_path(root, mode="analysis")
    stats_second = runner.ingest_path(root, mode="analysis")

    assert stats_first.files_processed == 1
    assert stats_second.files_processed == 0


def test_ingest_parent_relationships(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    file_path = root / "mod.py"
    file_path.write_text(
        "class A:\n"
        "    def f(self):\n"
        "        if True:\n"
        "            return 1\n"
    )

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runs_dir = tmp_path / "runs"
    runner = IngestionRunner(db=db, runs_dir=runs_dir)
    runner.ingest_path(root, mode="analysis")

    units = list(db.iter_units(limit=50))
    qn_to_unit = {u["qualified_name"]: u for u in units if u.get("qualified_name")}
    assert "A" in qn_to_unit or "mod.A" in qn_to_unit


def test_ingest_builds_import_call_use_edges(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mod.py").write_text(
        "import math\n"
        "def helper(x):\n"
        "    return math.floor(x)\n"
        "def runner(v):\n"
        "    return helper(v)\n",
        encoding="utf-8",
    )

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    runner.ingest_path(root, mode="analysis")

    rel_counts = db.relationship_counts()
    assert rel_counts.get("imports", 0) >= 1
    assert rel_counts.get("calls", 0) >= 1
    assert rel_counts.get("uses", 0) >= 1

    dep = db.module_dependency_graph(rel_types=["imports", "calls", "uses"], limit_edges=2000)
    assert dep["summary"]["relationship_rows"] >= 1
    assert dep["summary"]["edge_count"] >= 1
