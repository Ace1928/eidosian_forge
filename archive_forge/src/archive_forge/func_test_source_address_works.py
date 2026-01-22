import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_source_address_works(self):
    expected_port = _get_free_port(self.host)

    def request_handler(listener):
        sock = listener.accept()[0]
        assert sock.getpeername()[0] == '127.0.0.1'
        assert sock.getpeername()[1] == expected_port
        handler = handle_socks5_negotiation(sock, negotiate=False)
        addr, port = next(handler)
        assert addr == '16.17.18.19'
        assert port == 80
        handler.send(True)
        while True:
            buf = sock.recv(65535)
            if buf.endswith(b'\r\n\r\n'):
                break
        sock.sendall(b'HTTP/1.1 200 OK\r\nServer: SocksTestServer\r\nContent-Length: 0\r\n\r\n')
        sock.close()
    self._start_server(request_handler)
    proxy_url = 'socks5://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url, source_address=('127.0.0.1', expected_port)) as pm:
        response = pm.request('GET', 'http://16.17.18.19')
        assert response.status == 200