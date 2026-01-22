import threading
import time
import pytest
from .._compat import IS_MACOS, IS_WINDOWS  # noqa: WPS436
from ..server import Gateway, HTTPServer
from ..testing import (  # noqa: F401  # pylint: disable=unused-import
from ..testing import get_server_client
def make_http_server(bind_addr):
    """Create and start an HTTP server bound to ``bind_addr``."""
    httpserver = HTTPServer(bind_addr=bind_addr, gateway=Gateway)
    threading.Thread(target=httpserver.safe_start).start()
    while not httpserver.ready:
        time.sleep(0.1)
    return httpserver