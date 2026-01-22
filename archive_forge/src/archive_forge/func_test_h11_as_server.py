import json
import os.path
import socket
import socketserver
import threading
from contextlib import closing, contextmanager
from http.server import SimpleHTTPRequestHandler
from typing import Callable, Generator
from urllib.request import urlopen
import h11
def test_h11_as_server() -> None:
    with socket_server(H11RequestHandler) as httpd:
        host, port = httpd.server_address
        url = 'http://{}:{}/some-path'.format(host, port)
        with closing(urlopen(url)) as f:
            assert f.getcode() == 200
            data = f.read()
    info = json.loads(data.decode('ascii'))
    print(info)
    assert info['method'] == 'GET'
    assert info['target'] == '/some-path'
    assert 'urllib' in info['headers']['user-agent']