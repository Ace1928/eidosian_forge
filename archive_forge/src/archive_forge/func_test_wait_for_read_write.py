import signal
import socket
import threading
import time
import pytest
from urllib3.util.wait import (
from .socketpair_helper import socketpair
def test_wait_for_read_write(spair):
    a, b = spair
    assert not wait_for_read(a, 0)
    assert wait_for_write(a, 0)
    b.send(b'x')
    assert wait_for_read(a, 0)
    assert wait_for_write(a, 0)
    a.setblocking(False)
    try:
        while True:
            a.send(b'x' * 999999)
    except (OSError, socket.error):
        pass
    assert not wait_for_write(a, 0)