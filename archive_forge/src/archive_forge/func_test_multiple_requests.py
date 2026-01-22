import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_multiple_requests(self):
    """multiple requests can be served"""
    requests_to_handle = 5
    server = Server.basic_response_server(requests_to_handle=requests_to_handle)
    with server as (host, port):
        server_url = 'http://{}:{}'.format(host, port)
        for _ in range(requests_to_handle):
            r = requests.get(server_url)
            assert r.status_code == 200
        with pytest.raises(requests.exceptions.ConnectionError):
            r = requests.get(server_url)