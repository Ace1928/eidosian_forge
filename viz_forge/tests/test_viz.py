"""Tests for viz_forge core functionality."""

import pytest


class TestVizForgeCore:
    """Test viz_forge basic functionality."""

    def test_import(self):
        """Test that viz_forge can be imported."""
        import viz_forge
        assert hasattr(viz_forge, "__version__")

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from viz_forge.cli import main
        assert callable(main)


class TestVizForgeVersion:
    """Test version information."""

    def test_version_format(self):
        """Test version string format."""
        from viz_forge import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
