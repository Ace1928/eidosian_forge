import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_fragment_update_on_redirect():
    """Verify we only append previous fragment if one doesn't exist on new
    location. If a new fragment is encountered in a Location header, it should
    be added to all subsequent requests.
    """

    def response_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        sock.send(b'HTTP/1.1 302 FOUND\r\nContent-Length: 0\r\nLocation: /get#relevant-section\r\n\r\n')
        consume_socket_content(sock, timeout=0.5)
        sock.send(b'HTTP/1.1 302 FOUND\r\nContent-Length: 0\r\nLocation: /final-url/\r\n\r\n')
        consume_socket_content(sock, timeout=0.5)
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')
    close_server = threading.Event()
    server = Server(response_handler, wait_to_close_event=close_server)
    with server as (host, port):
        url = 'http://{}:{}/path/to/thing/#view=edit&token=hunter2'.format(host, port)
        r = requests.get(url)
        raw_request = r.content
        assert r.status_code == 200
        assert len(r.history) == 2
        assert r.history[0].request.url == url
        assert r.history[1].request.url == 'http://{}:{}/get#relevant-section'.format(host, port)
        assert r.url == 'http://{}:{}/final-url/#relevant-section'.format(host, port)
        close_server.set()