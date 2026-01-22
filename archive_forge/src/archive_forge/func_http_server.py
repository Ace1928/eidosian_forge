import threading
import time
import pytest
from .._compat import IS_MACOS, IS_WINDOWS  # noqa: WPS436
from ..server import Gateway, HTTPServer
from ..testing import (  # noqa: F401  # pylint: disable=unused-import
from ..testing import get_server_client
@pytest.fixture
def http_server():
    """Provision a server creator as a fixture."""

    def start_srv():
        bind_addr = (yield)
        if bind_addr is None:
            return
        httpserver = make_http_server(bind_addr)
        yield httpserver
        yield httpserver
    srv_creator = iter(start_srv())
    next(srv_creator)
    yield srv_creator
    try:
        while True:
            httpserver = next(srv_creator)
            if httpserver is not None:
                httpserver.stop()
    except StopIteration:
        pass