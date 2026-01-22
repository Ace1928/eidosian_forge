import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def sleep_until(self):
    t = self.timers
    if not t:
        return None
    return t[0][0]