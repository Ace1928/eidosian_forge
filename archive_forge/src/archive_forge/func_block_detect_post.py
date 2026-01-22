import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def block_detect_post(self):
    if hasattr(self, '_old_signal_handler') and self._old_signal_handler:
        signal.signal(signal.SIGALRM, self._old_signal_handler)
    signal.alarm(0)