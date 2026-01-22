import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_server_closes(self):
    """the server closes when leaving the context manager"""
    with Server.basic_response_server() as (host, port):
        sock = socket.socket()
        sock.connect((host, port))
        sock.close()
    with pytest.raises(socket.error):
        new_sock = socket.socket()
        new_sock.connect((host, port))