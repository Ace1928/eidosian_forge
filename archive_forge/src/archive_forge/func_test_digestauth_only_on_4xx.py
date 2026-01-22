import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/psf/requests/issues/3772.
    """
    text_200_chal = b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nWWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d", opaque="372825293d1c26955496c80ed6426e9e", realm="me@kennethreitz.com", qop=auth\r\n\r\n'
    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b'GET / HTTP/1.1')
        sock.send(text_200_chal)
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''
        return request_content
    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)
    with server as (host, port):
        url = 'http://{}:{}/'.format(host, port)
        r = requests.get(url, auth=auth)
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()