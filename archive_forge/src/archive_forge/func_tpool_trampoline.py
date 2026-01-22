import atexit
import os
import sys
import traceback
import eventlet
from eventlet import event, greenio, greenthread, patcher, timeout
def tpool_trampoline():
    global _rspq
    while True:
        try:
            _c = _rsock.recv(1)
            assert _c
        except ValueError:
            break
        while not _rspq.empty():
            try:
                e, rv = _rspq.get(block=False)
                e.send(rv)
                e = rv = None
            except Empty:
                pass