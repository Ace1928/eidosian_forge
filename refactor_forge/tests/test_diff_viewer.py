from __future__ import annotations

from pathlib import Path

from refactor_forge.diff_viewer import load_and_diff, unified_diff_text


def test_unified_diff_text_contains_expected_markers() -> None:
    diff = unified_diff_text("a = 1\n", "a = 2\n", fromfile="old.py", tofile="new.py")
    assert "--- old.py" in diff
    assert "+++ new.py" in diff
    assert "-a = 1" in diff
    assert "+a = 2" in diff


def test_load_and_diff_reads_files(tmp_path: Path) -> None:
    src = tmp_path / "src.py"
    proposed = tmp_path / "proposed.py"
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    proposed.write_text("def f():\n    return 2\n", encoding="utf-8")

    diff = load_and_diff(src, proposed)
    assert str(src) in diff
    assert str(proposed) in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff
