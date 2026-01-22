import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def multiple_content_length_response_handler(sock):
    request_content = consume_socket_content(sock, timeout=0.5)
    sock.send(b'HTTP/1.1 200 OK\r\n' + b'Content-Type: text/plain\r\n' + b'Content-Length: 16\r\n' + b'Content-Length: 32\r\n\r\n' + b'-- Bad Actor -- Original Content\r\n')
    return request_content