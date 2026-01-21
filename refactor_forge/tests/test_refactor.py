import pytest
from refactor_forge import RefactorForge

def test_rename():
    rf = RefactorForge()
    source = "def foo():\n    return foo"
    result = rf.transform(source, rename_map={"foo": "bar"})
    assert "def bar():" in result
    assert "return bar" in result

def test_remove_docs():
    rf = RefactorForge()
    source = 'def foo():\n    """Docstring."""\n    pass'
    result = rf.transform(source, remove_docs=True)
    assert '"""Docstring."""' not in result
    assert "pass" in result
