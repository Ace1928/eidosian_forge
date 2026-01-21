import pytest
from file_forge import FileForge

def test_file_forge_basic(tmp_path):
    ff = FileForge(base_path=tmp_path)
    
    # Create structure
    structure = {
        "src": {
            "main.py": "print('hello')",
            "utils.py": "def foo(): pass"
        },
        "docs": {
            "readme.md": "# Readme"
        }
    }
    ff.ensure_structure(structure)
    
    assert (tmp_path / "src" / "main.py").exists()
    assert (tmp_path / "docs" / "readme.md").read_text() == "# Readme"

    # Search Content
    matches = ff.search_content("print", directory=tmp_path)
    assert len(matches) == 1
    assert matches[0].name == "main.py"

    # Find Files (Glob)
    py_files = ff.find_files("*.py", directory=tmp_path)
    assert len(py_files) == 2

    # Duplicates
    (tmp_path / "src" / "copy.py").write_text("print('hello')")
    dups = ff.find_duplicates(tmp_path)
    assert len(dups) == 1
    assert len(list(dups.values())[0]) == 2