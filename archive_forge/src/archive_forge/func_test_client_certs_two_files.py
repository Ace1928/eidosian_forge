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
def test_client_certs_two_files(self):
    """
        Having a client cert in a separate file to its associated key works
        properly.
        """
    done_receiving = Event()
    client_certs = []

    def socket_handler(listener):
        sock = listener.accept()[0]
        sock = self._wrap_in_ssl(sock)
        client_certs.append(sock.getpeercert())
        data = b''
        while not data.endswith(b'\r\n\r\n'):
            data += sock.recv(8192)
        sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
        done_receiving.wait(5)
        sock.close()
    self._start_server(socket_handler)
    with HTTPSConnectionPool(self.host, self.port, cert_file=self.cert_path, key_file=self.key_path, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
        pool.request('GET', '/', retries=0)
        done_receiving.set()
        assert len(client_certs) == 1