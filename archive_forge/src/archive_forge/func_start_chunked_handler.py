import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def start_chunked_handler(self):
    self.buffer = b''

    def socket_handler(listener):
        sock = listener.accept()[0]
        while not self.buffer.endswith(b'\r\n0\r\n\r\n'):
            self.buffer += sock.recv(65536)
        sock.send(b'HTTP/1.1 200 OK\r\nContent-type: text/plain\r\nContent-Length: 0\r\n\r\n')
        sock.close()
    self._start_server(socket_handler)