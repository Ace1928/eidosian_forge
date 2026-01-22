import signal
import socket
import threading
import time
import pytest
from urllib3.util.wait import (
from .socketpair_helper import socketpair
@pytest.mark.skipif(not hasattr(signal, 'setitimer'), reason='need setitimer() support')
@pytest.mark.parametrize('wfs', variants)
def test_eintr_zero_timeout(wfs, spair):
    a, b = spair
    interrupt_count = [0]

    def handler(sig, frame):
        assert sig == signal.SIGALRM
        interrupt_count[0] += 1
    old_handler = signal.signal(signal.SIGALRM, handler)
    try:
        assert not wfs(a, read=True, timeout=0)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.001, 0.001)
            for i in range(100000):
                wfs(a, read=True, timeout=0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    assert interrupt_count[0] > 0