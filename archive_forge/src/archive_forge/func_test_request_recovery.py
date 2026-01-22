import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
@pytest.mark.skip(reason='this fails non-deterministically under pytest-xdist')
def test_request_recovery(self):
    """can check the requests content"""
    server = Server.basic_response_server(requests_to_handle=2)
    first_request = b'put your hands up in the air'
    second_request = b'put your hand down in the floor'
    with server as address:
        sock1 = socket.socket()
        sock2 = socket.socket()
        sock1.connect(address)
        sock1.sendall(first_request)
        sock1.close()
        sock2.connect(address)
        sock2.sendall(second_request)
        sock2.close()
    assert server.handler_results[0] == first_request
    assert server.handler_results[1] == second_request