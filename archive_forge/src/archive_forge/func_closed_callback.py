import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def closed_callback(fileno):
    """ Used to de-fang a callback that may be triggered by a loop in BaseHub.wait
    """
    pass