"""Web Interface Forge - WebSocket server/client for Eidosian Forge."""

__version__ = "0.1.0"

from . import eidos_server
from . import eidos_client

__all__ = ["__version__", "eidos_server", "eidos_client"]
