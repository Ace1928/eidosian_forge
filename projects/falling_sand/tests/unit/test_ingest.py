import json
from pathlib import Path

from falling_sand import indexer
from falling_sand import ingest


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_ingest_index(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    document = indexer.index_project(source_root, tests_root)
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(document.to_dict(), indent=2), encoding="utf-8")

    db_path = tmp_path / "index.db"
    run_id = ingest.ingest_index(index_path, db_path)

    assert run_id > 0
    assert db_path.exists()


def test_build_parser_defaults() -> None:
    parser = ingest.build_parser()
    args = parser.parse_args(["--index", "artifacts/index.json"])
    assert args.db == Path("artifacts/index.db")
    assert args.batch_size == 1000
