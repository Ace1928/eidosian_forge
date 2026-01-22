import signal
import socket
import threading
import time
import pytest
from urllib3.util.wait import (
from .socketpair_helper import socketpair
@pytest.mark.parametrize('wfs', variants)
def test_wait_for_socket(wfs, spair):
    a, b = spair
    with pytest.raises(RuntimeError):
        wfs(a, read=False, write=False)
    assert not wfs(a, read=True, timeout=0)
    assert wfs(a, write=True, timeout=0)
    b.send(b'x')
    assert wfs(a, read=True, timeout=0)
    assert wfs(a, read=True, timeout=10)
    assert wfs(a, read=True, timeout=None)
    a.setblocking(False)
    try:
        while True:
            a.send(b'x' * 999999)
    except (OSError, socket.error):
        pass
    assert not wfs(a, write=True, timeout=0)
    assert wfs(a, read=True, write=True, timeout=0)
    assert a.recv(1) == b'x'
    assert not wfs(a, read=True, write=True, timeout=0)
    b.close()
    assert wfs(a, read=True, timeout=0)
    with pytest.raises(Exception):
        wfs(b, read=True)