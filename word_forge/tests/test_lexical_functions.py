from pathlib import Path

from word_forge.parser.lexical_functions import (
    file_exists,
    read_json_file,
    read_jsonl_file,
    safe_open,
)


def test_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    assert not file_exists(path)
    path.write_text("data")
    assert file_exists(path)


def test_safe_open_missing(tmp_path: Path) -> None:
    with safe_open(tmp_path / "missing.txt") as handle:
        assert handle is None


def test_read_json_file_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert read_json_file(path, {"d": 1}) == {"d": 1}


def test_read_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')

    def proc(data):
        return data["a"]

    assert read_jsonl_file(path, proc) == [1, 2]


def test_read_jsonl_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    assert read_jsonl_file(path, lambda data: data) == []
