from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.librarian.core import CodeLibrarian


def test_analyzer(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("class Foo:\n    def bar(self):\n        return 1\n")

    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)

    assert len(res["classes"]) == 1
    assert res["classes"][0]["name"] == "Foo"
    assert "bar" in res["classes"][0]["methods"]
    assert res["classes"][0]["line_start"] == 1
    assert res["functions"] == []
    assert res["module"]["line_start"] == 1


def test_librarian(tmp_path):
    lib_path = tmp_path / "lib.json"
    lib = CodeLibrarian(lib_path)

    sid = lib.add_snippet("print('hello')", metadata={"type": "python"})
    assert sid

    res = lib.search("print")
    assert len(res) == 1
    assert res[0]["metadata"]["type"] == "python"


def test_analyzer_function_spans(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text("def add(a, b):\n" "    return a + b\n" "\n" "def sub(a, b):\n" "    return a - b\n")

    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)

    assert len(res["functions"]) == 2
    first = res["functions"][0]
    second = res["functions"][1]
    assert first["name"] == "add"
    assert first["line_start"] == 1
    assert first["line_end"] == 2
    assert second["name"] == "sub"
    assert second["line_start"] == 4
    assert second["line_end"] == 5


def test_analyzer_nodes_and_complexity(tmp_path):
    f = tmp_path / "logic.py"
    f.write_text("def decision(x):\n" "    if x > 0:\n" "        return 1\n" "    else:\n" "        return -1\n")

    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)
    nodes = res["nodes"]

    func_nodes = [n for n in nodes if n["unit_type"] == "function"]
    if_nodes = [n for n in nodes if n["unit_type"] == "if_block"]

    assert func_nodes
    assert if_nodes
    assert func_nodes[0]["complexity"] >= 2
    assert func_nodes[0]["parent_qualified_name"] is None
