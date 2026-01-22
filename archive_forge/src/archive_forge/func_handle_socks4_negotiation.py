import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def handle_socks4_negotiation(sock, username=None):
    """
    Handle the SOCKS4 handshake.

    Returns a generator object that allows us to break the handshake into
    steps so that the test code can intervene at certain useful points.
    """
    received_version = sock.recv(1)
    command = sock.recv(1)
    port = _read_exactly(sock, 2)
    port = (ord(port[0:1]) << 8) + ord(port[1:2])
    addr = _read_exactly(sock, 4)
    provided_username = _read_until(sock, b'\x00')[:-1]
    if addr == b'\x00\x00\x00\x01':
        addr = _read_until(sock, b'\x00')[:-1]
    else:
        addr = socket.inet_ntoa(addr)
    assert received_version == SOCKS_VERSION_SOCKS4
    assert command == b'\x01'
    if username is not None and username != provided_username:
        sock.sendall(b'\x00]\x00\x00\x00\x00\x00\x00')
        sock.close()
        yield False
        return
    succeed = (yield (addr, port))
    if succeed:
        response = b'\x00Z\xea`\x7f\x00\x00\x01'
    else:
        response = b'\x00[\x00\x00\x00\x00\x00\x00'
    sock.sendall(response)
    yield True