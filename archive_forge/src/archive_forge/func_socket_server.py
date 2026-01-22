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
@contextmanager
def socket_server(handler: Callable[..., socketserver.BaseRequestHandler]) -> Generator[socketserver.TCPServer, None, None]:
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.01})
    thread.daemon = True
    try:
        thread.start()
        yield httpd
    finally:
        httpd.shutdown()