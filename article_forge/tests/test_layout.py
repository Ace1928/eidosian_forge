from pathlib import Path


def test_packaging_layout():
    """Validate article_forge package structure."""
    root_dir = Path(__file__).resolve().parents[1]
    assert (root_dir / "pyproject.toml").exists(), "pyproject.toml required"
    # Modern Python packaging uses src/ layout, no root __init__.py
    src_pkg = root_dir / "src" / root_dir.name
    assert src_pkg.exists(), f"src/{root_dir.name}/ must exist"
    assert (src_pkg / "__init__.py").exists() or any(src_pkg.rglob("*.py")), "Package must have Python files"
