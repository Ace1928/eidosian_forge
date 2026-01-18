import json
from pathlib import Path

import pytest

from falling_sand import indexer


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_iter_python_files_excludes_dirs(tmp_path: Path) -> None:
    keep = tmp_path / "keep.py"
    skip = tmp_path / ".venv" / "skip.py"
    _write_file(keep, "def keep():\n    return 1\n")
    _write_file(skip, "def skip():\n    return 2\n")

    files = list(indexer.iter_python_files(tmp_path, exclude_dirs=(".venv",)))
    assert keep in files
    assert skip not in files


def test_iter_python_files_skips_non_py(tmp_path: Path) -> None:
    keep = tmp_path / "keep.py"
    skip = tmp_path / "skip.txt"
    _write_file(keep, "def keep():\n    return 1\n")
    _write_file(skip, "not python\n")

    files = list(indexer.iter_python_files(tmp_path, exclude_dirs=()))
    assert keep in files
    assert skip not in files


def test_parse_python_file_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    _write_file(bad, "def broken(:\n")

    with pytest.raises(ValueError, match="Syntax error"):
        indexer.parse_python_file(bad)


def test_format_signature_simple() -> None:
    node = indexer.parse_python_file(Path(__file__)).body[0]
    assert isinstance(node, indexer.ast.AST)

    func = indexer.ast.parse("def demo(a, b=1, *, c=2):\n    return a\n").body[0]
    signature = indexer.format_signature(func)
    assert signature is not None
    assert "a" in signature


def test_module_name_from_path(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    path = root / "module.py"
    _write_file(path, "def demo():\n    return 1\n")

    assert indexer.module_name_from_path(path, root) == "module"


def test_module_name_from_init(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    path = root / "__init__.py"
    _write_file(path, "def demo():\n    return 1\n")

    assert indexer.module_name_from_path(path, root) == "__init__"


def test_extract_definitions(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    _write_file(
        path,
        """
class Demo:
    def run(self):
        return 1

def helper(x):
    return x
""".lstrip(),
    )
    tree = indexer.parse_python_file(path)
    entries = indexer.extract_definitions(tree, "mod", str(path), "source")

    kinds = {(entry.kind, entry.qualname) for entry in entries}
    assert ("class", "Demo") in kinds
    assert ("method", "Demo.run") in kinds
    assert ("function", "helper") in kinds


def test_index_root(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    _write_file(path, "def demo():\n    return 1\n")

    entries = indexer.index_root(tmp_path, "source", exclude_dirs=())
    assert len(entries) == 1
    assert entries[0].name == "demo"


def test_index_project_stats(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    result = indexer.index_project(source_root, tests_root)

    assert result.stats["total"] == 2
    assert result.stats["source"] == 1
    assert result.stats["test"] == 1


def test_build_parser_defaults() -> None:
    parser = indexer.build_parser()
    args = parser.parse_args([])
    assert args.source_root == Path("src")
    assert args.tests_root == Path("tests")
    assert args.output is None
    assert args.test_report == []
    assert args.profile_stats is None
    assert args.profile_top_n == 20
    assert args.benchmark_report == []
    assert args.exclude_dir == []
    assert args.allow_missing_tests is False


def test_main_writes_output(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")
    output = tmp_path / "index.json"

    exit_code = indexer.main(
        [
            "--source-root",
            str(source_root),
            "--tests-root",
            str(tests_root),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stats"]["total"] == 2
    assert payload["schema_version"] >= 2
    assert payload["test_summary"] is None


def test_normalize_exclude_dirs_deduplicates() -> None:
    result = indexer.normalize_exclude_dirs([".venv", ".git", ".venv"])
    assert result.count(".venv") == 1
    assert ".git" in result


def test_validate_root_allow_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    indexer.validate_root(missing, allow_missing=True)


def test_index_project_allow_missing_tests(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir(parents=True)
    (source_root / "pkg.py").write_text("def demo():\n    return 1\n", encoding="utf-8")
    tests_root = tmp_path / "tests"

    result = indexer.index_project(source_root, tests_root, allow_missing_tests=True)

    assert result.stats["source"] == 1
    assert result.stats["test"] == 0
