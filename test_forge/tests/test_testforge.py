"""Tests for test_forge core functionality."""

import pytest


class TestTestForgeCore:
    """Test test_forge basic functionality."""

    def test_import(self):
        """Test that test_forge can be imported."""
        import test_forge
        assert hasattr(test_forge, "__version__")

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from test_forge.cli import main
        assert callable(main)


class TestTestForgeVersion:
    """Test version information."""

    def test_version_format(self):
        """Test version string format."""
        from test_forge import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2


class TestTestForgeStructure:
    """Test test_forge package structure."""

    def test_module_path(self):
        """Test module has correct path."""
        import test_forge
        assert "test_forge" in test_forge.__file__

    def test_cli_module_exists(self):
        """Test CLI module exists."""
        from test_forge import cli
        assert hasattr(cli, "main")
