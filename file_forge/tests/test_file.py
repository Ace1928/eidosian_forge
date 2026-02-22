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


def test_search_content_uses_ripgrep_when_available(tmp_path, monkeypatch):
    ff = FileForge(base_path=tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("needle", encoding="utf-8")

    monkeypatch.setattr("file_forge.core.shutil.which", lambda _: "/usr/bin/rg")

    class _Result:
        returncode = 0
        stdout = f"{target}\n"

    def _fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr("file_forge.core.subprocess.run", _fake_run)
    matches = ff.search_content("needle", directory=tmp_path)
    assert matches == [target]


def test_search_content_falls_back_when_ripgrep_fails(tmp_path, monkeypatch):
    ff = FileForge(base_path=tmp_path)
    target = tmp_path / "b.txt"
    target.write_text("fallback-needle", encoding="utf-8")

    monkeypatch.setattr("file_forge.core.shutil.which", lambda _: "/usr/bin/rg")

    def _raise_run(*args, **kwargs):
        raise OSError("rg unavailable")

    monkeypatch.setattr("file_forge.core.subprocess.run", _raise_run)
    matches = ff.search_content("fallback-needle", directory=tmp_path)
    assert matches == [target]
