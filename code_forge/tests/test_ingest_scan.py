from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def test_scan_respects_extensions(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('a')\n")
    (root / "b.txt").write_text("ignore")

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")

    files, loc, total_bytes = runner._scan_files(root, {".py"}, [], max_files=None)
    assert len(files) == 1
    assert files[0][0].name == "a.py"
    assert loc >= 1
    assert total_bytes > 0
