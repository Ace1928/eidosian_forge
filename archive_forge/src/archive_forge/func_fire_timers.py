import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def fire_timers(self, when):
    t = self.timers
    heappop = heapq.heappop
    while t:
        next = t[0]
        exp = next[0]
        timer = next[1]
        if when < exp:
            break
        heappop(t)
        try:
            if timer.called:
                self.timers_canceled -= 1
            else:
                timer()
        except self.SYSTEM_EXCEPTIONS:
            raise
        except:
            self.squelch_timer_exception(timer, sys.exc_info())