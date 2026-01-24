"""Tests for mkey_forge (mnemonic key generation)."""

import pytest


class TestMkeyForgeCore:
    """Test mkey_forge basic functionality."""

    def test_import(self):
        """Test that mkey_forge can be imported."""
        import mkey_forge
        assert hasattr(mkey_forge, "__version__")

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from mkey_forge.cli import main
        assert callable(main)


class TestMkeyGeneration:
    """Test mnemonic key generation."""

    def test_mkey_module_exists(self):
        """Test mkey module can be imported."""
        from mkey_forge import mkey_generator
        assert mkey_generator is not None

    def test_mkey_generator_class(self):
        """Test mkey_generator class exists."""
        from mkey_forge.mkey import mkey_generator
        assert callable(mkey_generator)

    def test_main_function(self):
        """Test main function exists and is callable."""
        from mkey_forge.mkey import main
        assert callable(main)

