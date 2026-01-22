from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
@pytest.fixture
def native_server():
    """Set up and tear down a Cheroot HTTP server instance."""
    with cheroot_server(cheroot.server.HTTPServer) as srv:
        yield srv