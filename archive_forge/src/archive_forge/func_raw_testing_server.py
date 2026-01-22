import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
@pytest.fixture
def raw_testing_server(wsgi_server_client):
    """Attach a WSGI app to the given server and preconfigure it."""
    app = Controller()

    def _timeout(req, resp):
        return str(wsgi_server.timeout)
    app.handlers['/timeout'] = _timeout
    wsgi_server = wsgi_server_client.server_instance
    wsgi_server.wsgi_app = app
    wsgi_server.max_request_body_size = 1001
    wsgi_server.timeout = timeout
    wsgi_server.server_client = wsgi_server_client
    wsgi_server.keep_alive_conn_limit = 2
    return wsgi_server