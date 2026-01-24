"""Tests for prompt_forge core functionality."""

import pytest


class TestPromptForgeCore:
    """Test prompt_forge basic functionality."""

    def test_import(self):
        """Test that prompt_forge can be imported."""
        import prompt_forge
        assert hasattr(prompt_forge, "__version__")

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from prompt_forge.cli import main
        assert callable(main)


class TestPromptForgeVersion:
    """Test version information."""

    def test_version_format(self):
        """Test version string format."""
        from prompt_forge import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
