import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_fragment_not_sent_with_request():
    """Verify that the fragment portion of a URI isn't sent to the server."""

    def response_handler(sock):
        req = consume_socket_content(sock, timeout=0.5)
        sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: ' + bytes(len(req)) + b'\r\n\r\n' + req)
    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)
    with server as (host, port):
        url = 'http://{}:{}/path/to/thing/#view=edit&token=hunter2'.format(host, port)
        r = requests.get(url)
        raw_request = r.content
        assert r.status_code == 200
        headers, body = raw_request.split(b'\r\n\r\n', 1)
        status_line, headers = headers.split(b'\r\n', 1)
        assert status_line == b'GET /path/to/thing/ HTTP/1.1'
        for frag in (b'view', b'edit', b'token', b'hunter2'):
            assert frag not in headers
            assert frag not in body
        close_server.set()