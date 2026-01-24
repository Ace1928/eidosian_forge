"""Tests for sms_forge core functionality."""

import pytest


class TestSmsForgeCore:
    """Test sms_forge basic functionality."""

    def test_import(self):
        """Test that sms_forge can be imported."""
        import sms_forge
        assert hasattr(sms_forge, "__version__")

    def test_placeholder_function(self):
        """Test placeholder function exists."""
        from sms_forge import placeholder
        with pytest.raises(NotImplementedError):
            placeholder()

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from sms_forge.cli import main
        assert callable(main)
