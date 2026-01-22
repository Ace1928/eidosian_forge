import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_text_response(self):
    """the text_response_server sends the given text"""
    server = Server.text_response_server('HTTP/1.1 200 OK\r\n' + 'Content-Length: 6\r\n' + '\r\nroflol')
    with server as (host, port):
        r = requests.get('http://{}:{}'.format(host, port))
        assert r.status_code == 200
        assert r.text == u'roflol'
        assert r.headers['Content-Length'] == '6'