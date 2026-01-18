from pathlib import Path

from scripts import profile_index


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_parser_defaults() -> None:
    parser = profile_index.build_parser()
    args = parser.parse_args([])
    assert args.source_root == Path("src")
    assert args.tests_root == Path("tests")
    assert args.output == Path("artifacts/profile.pstats")


def test_profile_index_writes_output(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    output = tmp_path / "profile.pstats"
    result = profile_index.profile_index(source_root, tests_root, output)

    assert result == output
    assert output.exists()


def test_main(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    output = tmp_path / "profile.pstats"
    exit_code = profile_index.main(
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
    assert output.exists()
