from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
def test_timeout_errors_cause_retries(self):

    def socket_handler(listener):
        sock_timeout = listener.accept()[0]
        sock = listener.accept()[0]
        sock_timeout.close()
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf += sock.recv(65536)
        body = 'Response 2'
        sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
        sock.close()
    default_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(1)
    try:
        self._start_server(socket_handler)
        t = Timeout(connect=LONG_TIMEOUT, read=LONG_TIMEOUT)
        with HTTPConnectionPool(self.host, self.port, timeout=t) as pool:
            response = pool.request('GET', '/', retries=1)
            assert response.status == 200
            assert response.data == b'Response 2'
    finally:
        socket.setdefaulttimeout(default_timeout)