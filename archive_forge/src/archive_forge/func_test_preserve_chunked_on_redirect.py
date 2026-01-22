import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def test_preserve_chunked_on_redirect(self, monkeypatch):
    self.chunked_requests = 0

    def socket_handler(listener):
        for i in range(2):
            sock = listener.accept()[0]
            request = ConnectionMarker.consume_request(sock)
            if b'Transfer-Encoding: chunked' in request.split(b'\r\n'):
                self.chunked_requests += 1
            if i == 0:
                sock.sendall(b'HTTP/1.1 301 Moved Permanently\r\nLocation: /redirect\r\n\r\n')
            else:
                sock.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
            sock.close()
    self._start_server(socket_handler)
    with ConnectionMarker.mark(monkeypatch):
        with HTTPConnectionPool(self.host, self.port) as pool:
            retries = Retry(redirect=1)
            pool.urlopen('GET', '/', chunked=True, preload_content=False, retries=retries)
    assert self.chunked_requests == 2