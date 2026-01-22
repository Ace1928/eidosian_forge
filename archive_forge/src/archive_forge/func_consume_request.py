import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@classmethod
def consume_request(cls, sock, chunks=65536):
    """
        Consume a socket until after the HTTP request is sent.
        """
    consumed = bytearray()
    mark = cls._get_socket_mark(sock, True)
    while True:
        b = sock.recv(chunks)
        if not b:
            break
        consumed += b
        if consumed.endswith(mark):
            break
    return consumed