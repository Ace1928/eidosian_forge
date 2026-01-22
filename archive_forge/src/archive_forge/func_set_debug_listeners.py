import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def set_debug_listeners(self, value):
    if value:
        self.lclass = DebugListener
    else:
        self.lclass = FdListener