from pathlib import Path

from code_forge.analyzer.python_analyzer import CodeAnalyzer


def test_analyzer_detects_lambdas_and_comprehensions(tmp_path: Path) -> None:
    f = tmp_path / "comp.py"
    f.write_text(
        "def foo(items):\n"
        "    squares = [x*x for x in items if x % 2 == 0]\n"
        "    mapping = {x: x+1 for x in items}\n"
        "    evens = (x for x in items if x % 2 == 0)\n"
        "    f = lambda y: y + 1\n"
        "    return squares, mapping, list(evens), f\n"
    )

    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)
    nodes = res["nodes"]

    types = {n["unit_type"] for n in nodes}
    assert "list_comp" in types
    assert "dict_comp" in types
    assert "gen_exp" in types
    assert "lambda" in types
    # BoolOp only appears when explicit boolean operators are used.


def test_analyzer_emits_import_call_use_edges(tmp_path: Path) -> None:
    f = tmp_path / "edges.py"
    f.write_text(
        "import math\n"
        "from os import path\n"
        "def helper(value):\n"
        "    return math.floor(value)\n"
        "def run(v):\n"
        "    return helper(v) + (1 if path.exists('x') else 0)\n"
    )

    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)
    edges = res["edges"]
    assert edges
    rel_types = {edge["rel_type"] for edge in edges}
    assert "imports" in rel_types
    assert "calls" in rel_types
    assert "uses" in rel_types
