import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def remove_descriptor(self, fileno):
    """ Completely remove all listeners for this fileno.  For internal use
        only."""
    listeners = []
    listeners.append(self.listeners[READ].get(fileno, noop))
    listeners.append(self.listeners[WRITE].get(fileno, noop))
    listeners.extend(self.secondaries[READ].get(fileno, ()))
    listeners.extend(self.secondaries[WRITE].get(fileno, ()))
    for listener in listeners:
        try:
            listener.cb(fileno)
        except Exception:
            self.squelch_generic_exception(sys.exc_info())
    self.listeners[READ].pop(fileno, None)
    self.listeners[WRITE].pop(fileno, None)
    self.secondaries[READ].pop(fileno, None)
    self.secondaries[WRITE].pop(fileno, None)