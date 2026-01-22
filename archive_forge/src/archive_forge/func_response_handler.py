import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def response_handler(sock):
    consume_socket_content(sock, timeout=0.5)
    sock.send(b'HTTP/1.1 302 FOUND\r\nContent-Length: 0\r\nLocation: /get#relevant-section\r\n\r\n')
    consume_socket_content(sock, timeout=0.5)
    sock.send(b'HTTP/1.1 302 FOUND\r\nContent-Length: 0\r\nLocation: /final-url/\r\n\r\n')
    consume_socket_content(sock, timeout=0.5)
    sock.send(b'HTTP/1.1 200 OK\r\n\r\n')