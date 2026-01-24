"""Tests for lyrics_forge core functionality."""

import pytest


class TestLyricsForgeCore:
    """Test lyrics_forge basic functionality."""

    def test_import(self):
        """Test that lyrics_forge can be imported."""
        import lyrics_forge
        assert hasattr(lyrics_forge, "__version__")

    def test_placeholder_function(self):
        """Test placeholder function exists."""
        from lyrics_forge import placeholder
        with pytest.raises(NotImplementedError):
            placeholder()

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from lyrics_forge.cli import main
        assert callable(main)


class TestLyricsForgeVersion:
    """Test version information."""

    def test_version_format(self):
        """Test version string format."""
        from lyrics_forge import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
