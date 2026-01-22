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
def test_stream_none_unchunked_response_does_not_hang(self):
    done_event = Event()

    def socket_handler(listener):
        sock = listener.accept()[0]
        buf = b''
        while not buf.endswith(b'\r\n\r\n'):
            buf += sock.recv(65536)
        sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 12\r\nContent-type: text/plain\r\n\r\nhello, world')
        done_event.wait(5)
        sock.close()
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        r = pool.request('GET', '/', timeout=LONG_TIMEOUT, preload_content=False)
        assert [b'hello, world'] == list(r.stream(None))
        done_event.set()