import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def digest_response_handler(sock):
    request_content = consume_socket_content(sock, timeout=0.5)
    assert request_content.startswith(b'GET / HTTP/1.1')
    sock.send(text_200_chal)
    request_content = consume_socket_content(sock, timeout=0.5)
    assert request_content == b''
    return request_content