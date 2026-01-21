import pytest
from pathlib import Path
from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.librarian.core import CodeLibrarian

def test_analyzer(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("class Foo:\n    def bar(self): pass")
    
    analyzer = CodeAnalyzer()
    res = analyzer.analyze_file(f)
    
    assert len(res["classes"]) == 1
    assert res["classes"][0]["name"] == "Foo"
    assert "bar" in res["classes"][0]["methods"]

def test_librarian(tmp_path):
    lib_path = tmp_path / "lib.json"
    lib = CodeLibrarian(lib_path)
    
    sid = lib.add_snippet("print('hello')", metadata={"type": "python"})
    assert sid
    
    res = lib.search("print")
    assert len(res) == 1
    assert res[0]["metadata"]["type"] == "python"