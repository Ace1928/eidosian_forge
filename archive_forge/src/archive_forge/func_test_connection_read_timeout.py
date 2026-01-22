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
def test_connection_read_timeout(self):
    timed_out = Event()

    def socket_handler(listener):
        sock = listener.accept()[0]
        while not sock.recv(65536).endswith(b'\r\n\r\n'):
            pass
        timed_out.wait()
        sock.close()
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port, timeout=SHORT_TIMEOUT, retries=False, maxsize=3, block=True) as http:
        try:
            with pytest.raises(ReadTimeoutError):
                http.request('GET', '/', release_conn=False)
        finally:
            timed_out.set()
        assert http.pool.qsize() == http.pool.maxsize