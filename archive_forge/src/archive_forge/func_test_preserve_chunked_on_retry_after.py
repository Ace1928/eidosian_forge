import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def test_preserve_chunked_on_retry_after(self):
    self.chunked_requests = 0
    self.socks = []

    def socket_handler(listener):
        for _ in range(2):
            sock = listener.accept()[0]
            self.socks.append(sock)
            request = consume_socket(sock)
            if b'Transfer-Encoding: chunked' in request.split(b'\r\n'):
                self.chunked_requests += 1
            sock.send(b'HTTP/1.1 429 Too Many Requests\r\nContent-Type: text/plain\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
    self._start_server(socket_handler)
    with HTTPConnectionPool(self.host, self.port) as pool:
        retries = Retry(total=1)
        pool.urlopen('GET', '/', chunked=True, retries=retries)
        for sock in self.socks:
            sock.close()
    assert self.chunked_requests == 2