from pathlib import Path

from code_forge.library.db import CodeLibraryDB, CodeUnit
from code_forge.library.similarity import build_fingerprint, structural_hash


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


def test_duplicate_units_report_and_lookup(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    h = db.add_text("def shared():\n    return 42\n")
    u1 = CodeUnit(
        unit_type="function",
        name="shared",
        qualified_name="a.shared",
        file_path="a.py",
        content_hash=h,
    )
    u2 = CodeUnit(
        unit_type="function",
        name="shared",
        qualified_name="b.shared",
        file_path="b.py",
        content_hash=h,
    )
    db.add_unit(u1)
    db.add_unit(u2)

    groups = db.list_duplicate_units()
    assert len(groups) == 1
    assert groups[0]["occurrences"] == 2
    assert {u["qualified_name"] for u in groups[0]["units"]} == {"a.shared", "b.shared"}

    found = db.find_unit_by_qualified_name("a.shared")
    assert found is not None
    assert found["file_path"] == "a.py"


def test_trace_contains_bfs(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    mod = db.add_unit(CodeUnit(unit_type="module", name="m", qualified_name="m", file_path="m.py"))
    cls = db.add_unit(CodeUnit(unit_type="class", name="C", qualified_name="m.C", file_path="m.py"))
    fn = db.add_unit(CodeUnit(unit_type="method", name="f", qualified_name="m.C.f", file_path="m.py"))
    db.add_relationship(mod, cls, "contains")
    db.add_relationship(cls, fn, "contains")

    trace = db.trace_contains(mod, max_depth=3, max_nodes=20)
    assert trace["root"] == mod
    assert len(trace["nodes"]) == 3
    assert len(trace["edges"]) == 2

    children = db.get_children(mod)
    assert len(children) == 1
    assert children[0]["qualified_name"] == "m.C"

    parents = db.get_parents(fn)
    assert len(parents) == 1
    assert parents[0]["qualified_name"] == "m.C"
    assert db.count_units() == 3
    by_type = db.count_units_by_type()
    assert by_type["module"] == 1
    assert by_type["class"] == 1
    assert by_type["method"] == 1


def test_normalized_and_near_duplicates_and_semantic_search(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")

    src_a = "def compute_total(items):\n    total = sum(items)\n    return total\n"
    src_b = "def calculate_total(items):\n    total = sum(items)\n    return total\n"

    h_a = db.add_text(src_a)
    h_b = db.add_text(src_b)
    n_a, s_a, t_a = build_fingerprint(src_a)
    n_b, s_b, t_b = build_fingerprint(src_b)
    st_a = structural_hash(src_a)
    st_b = structural_hash(src_b)

    db.add_unit(
        CodeUnit(
            unit_type="function",
            name="compute_total",
            qualified_name="math.compute_total",
            file_path="src/a.py",
            content_hash=h_a,
            language="python",
            normalized_hash=n_a,
            structural_hash=st_a,
            simhash64=f"{s_a:016x}",
            token_count=t_a,
            semantic_text=src_a,
        )
    )
    db.add_unit(
        CodeUnit(
            unit_type="function",
            name="calculate_total",
            qualified_name="math.calculate_total",
            file_path="src/b.py",
            content_hash=h_b,
            language="python",
            normalized_hash=n_b,
            structural_hash=st_b,
            simhash64=f"{s_b:016x}",
            token_count=t_b,
            semantic_text=src_b,
        )
    )

    normalized = db.list_normalized_duplicates(min_occurrences=2, limit_groups=20)
    # Function name differs, so normalized content should remain distinct.
    assert normalized == []

    structural = db.list_structural_duplicates(min_occurrences=2, limit_groups=20)
    assert structural
    assert structural[0]["occurrences"] >= 2

    near = db.list_near_duplicates(max_hamming=32, min_token_count=1, limit_pairs=20)
    assert near
    pair = near[0]
    assert pair["left"]["file_path"] != pair["right"]["file_path"]

    matches = db.semantic_search("compute total items", limit=5, min_score=0.01)
    assert matches
    assert matches[0]["semantic_score"] >= 0.01


def test_file_metrics_and_language_counts(tmp_path: Path) -> None:
    db = CodeLibraryDB(tmp_path / "library.sqlite")
    h_py = db.add_text("def hi():\n    return 'hi'\n")
    h_js = db.add_text("function hi(){ return 'hi'; }\n")
    n_py, s_py, t_py = build_fingerprint("def hi():\n    return 'hi'\n")
    n_js, s_js, t_js = build_fingerprint("function hi(){ return 'hi'; }\n")
    st_py = structural_hash("def hi():\n    return 'hi'\n")
    st_js = structural_hash("function hi(){ return 'hi'; }\n")

    db.add_unit(
        CodeUnit(
            unit_type="function",
            name="hi",
            qualified_name="m.hi",
            file_path="src/m.py",
            language="python",
            content_hash=h_py,
            normalized_hash=n_py,
            structural_hash=st_py,
            simhash64=f"{s_py:016x}",
            token_count=t_py,
        )
    )
    db.add_unit(
        CodeUnit(
            unit_type="function",
            name="hi",
            qualified_name="web.hi",
            file_path="src/web.js",
            language="javascript",
            content_hash=h_js,
            normalized_hash=n_js,
            structural_hash=st_js,
            simhash64=f"{s_js:016x}",
            token_count=t_js,
        )
    )

    by_lang = db.count_units_by_language()
    assert by_lang["python"] == 1
    assert by_lang["javascript"] == 1

    metrics = db.file_metrics()
    assert any(rec["file_path"] == "src/m.py" for rec in metrics)
    assert any(rec["file_path"] == "src/web.js" for rec in metrics)
