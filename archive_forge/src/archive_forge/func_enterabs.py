import time
import heapq
from collections import namedtuple
from itertools import count
import threading
from time import monotonic as _time
def enterabs(self, time, priority, action, argument=(), kwargs=_sentinel):
    """Enter a new event in the queue at an absolute time.

        Returns an ID for the event which can be used to remove it,
        if necessary.

        """
    if kwargs is _sentinel:
        kwargs = {}
    with self._lock:
        event = Event(time, priority, next(self._sequence_generator), action, argument, kwargs)
        heapq.heappush(self._queue, event)
    return event