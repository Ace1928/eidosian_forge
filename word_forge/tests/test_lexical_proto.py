from pathlib import Path

from lexical_proto import file_exists, read_json_file, read_jsonl_file, safely_open_file


def test_file_exists(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    assert not file_exists(p)
    p.write_text("data")
    assert file_exists(p)


def test_safely_open_file_missing(tmp_path: Path) -> None:
    assert safely_open_file(tmp_path / "no.txt") is None


def test_read_json_file_invalid(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert read_json_file(p, {"d": 1}) == {"d": 1}


def test_read_jsonl_file(tmp_path: Path) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\ninvalid\n{"a":2}\n')

    def proc(d):
        return d["a"]

    assert read_jsonl_file(p, proc) == [1, 2]


def test_read_jsonl_file_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.jsonl"
    assert read_jsonl_file(p, lambda d: d) == []
