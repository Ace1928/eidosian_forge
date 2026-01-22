import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def timer_canceled(self, timer):
    self.timers_canceled += 1
    len_timers = len(self.timers) + len(self.next_timers)
    if len_timers > 1000 and len_timers / 2 <= self.timers_canceled:
        self.timers_canceled = 0
        self.timers = [t for t in self.timers if not t[1].called]
        self.next_timers = [t for t in self.next_timers if not t[1].called]
        heapq.heapify(self.timers)