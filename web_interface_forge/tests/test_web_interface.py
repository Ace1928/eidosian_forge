"""Tests for web_interface_forge core functionality."""

import pytest


class TestWebInterfaceForgeCore:
    """Test web_interface_forge basic functionality."""

    def test_import(self):
        """Test that web_interface_forge can be imported."""
        import web_interface_forge
        assert hasattr(web_interface_forge, "__version__")

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from web_interface_forge.cli import main
        assert callable(main)


class TestEidosServer:
    """Test Eidos server module."""

    def test_server_module_exists(self):
        """Test server module can be imported."""
        pytest.importorskip("playwright", reason="playwright optional for eidos_server")
        from web_interface_forge import eidos_server
        assert hasattr(eidos_server, "main")

    def test_helper_functions(self):
        """Test helper functions exist."""
        pytest.importorskip("playwright", reason="playwright optional for eidos_server")
        from web_interface_forge.eidos_server import sha256_text, safe_json
        assert callable(sha256_text)
        assert callable(safe_json)

    def test_sha256_text(self):
        """Test sha256_text function."""
        pytest.importorskip("playwright", reason="playwright optional for eidos_server")
        from web_interface_forge.eidos_server import sha256_text
        result = sha256_text("test")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest length


class TestEidosClient:
    """Test Eidos client module."""

    def test_client_module_exists(self):
        """Test client module can be imported."""
        from web_interface_forge import eidos_client
        assert hasattr(eidos_client, "main")

    def test_ts_str_function(self):
        """Test timestamp string function."""
        import time
        from web_interface_forge.eidos_client import ts_str
        result = ts_str(time.time())
        assert isinstance(result, str)
        # Contains date and time components
        assert len(result) > 10
