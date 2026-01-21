from pathlib import Path


def test_packaging_layout():
    root_dir = Path(__file__).resolve().parents[1]
    assert (root_dir / "pyproject.toml").exists()
    assert (root_dir / "__init__.py").exists()
    src_pkg = root_dir / "src" / root_dir.name
    assert src_pkg.exists()
    assert (src_pkg / "__init__.py").exists() or any(src_pkg.rglob("*.py"))
