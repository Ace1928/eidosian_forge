import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_is_socket_unix():
    with closing_socketpair(socket.AF_UNIX) as pair:
        for sock in pair:
            for arg in (sock, sock.fileno()):
                assert is_socket_unix(arg)
                assert not is_socket_unix(arg, path='/no/such/path')
                assert is_socket_unix(arg, socket.SOCK_STREAM)
                assert not is_socket_unix(arg, socket.SOCK_DGRAM)