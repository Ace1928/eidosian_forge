"""Tests for erais_forge core functionality."""

import pytest


class TestEraisForgeCore:
    """Test erais_forge basic functionality."""

    def test_import(self):
        """Test that erais_forge can be imported."""
        import erais_forge
        assert hasattr(erais_forge, "__version__")

    def test_placeholder_function(self):
        """Test placeholder function exists."""
        from erais_forge import placeholder
        with pytest.raises(NotImplementedError):
            placeholder()

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from erais_forge.cli import main
        assert callable(main)


class TestEraisForgeVersion:
    """Test version information."""

    def test_version_format(self):
        """Test version string format."""
        from erais_forge import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() or part == "0" for part in parts[:2])
