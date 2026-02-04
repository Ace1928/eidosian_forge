from pathlib import Path

from code_forge.library.db import CodeLibraryDB, CodeUnit


def test_add_text_dedup(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    content_hash_1 = db.add_text("print('hello')\n")
    content_hash_2 = db.add_text("print('hello')\n")

    assert content_hash_1 == content_hash_2
    assert db.get_text(content_hash_1) == "print('hello')\n"


def test_add_unit_roundtrip(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    content_hash = db.add_text("def add(a, b):\n    return a + b\n")

    unit = CodeUnit(
        unit_type="function",
        name="add",
        qualified_name="math.add",
        file_path="src/math.py",
        language="python",
        line_start=1,
        line_end=2,
        col_start=0,
        col_end=16,
        content_hash=content_hash,
    )

    unit_id = db.add_unit(unit)
    stored = db.get_unit(unit_id)

    assert stored is not None
    assert stored["name"] == "add"
    assert stored["qualified_name"] == "math.add"
    assert stored["content_hash"] == content_hash
    assert stored["line_start"] == 1
    assert stored["line_end"] == 2


def test_relationships(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    parent = CodeUnit(unit_type="module", name="math", file_path="src/math.py")
    child = CodeUnit(
        unit_type="function",
        name="add",
        file_path="src/math.py",
        qualified_name="math.add",
    )

    parent_id = db.add_unit(parent)
    child_id = db.add_unit(child)
    db.add_relationship(parent_id, child_id, "contains")
    db.add_relationship(parent_id, child_id, "contains")

    units = list(db.iter_units(limit=10))
    assert len(units) == 2
