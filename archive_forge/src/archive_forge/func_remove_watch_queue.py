from __future__ import annotations
import contextlib
import errno
import heapq
import logging
import os
import time
import typing
from itertools import count
import zmq
from .abstract_loop import EventLoop, ExitMainLoop
def remove_watch_queue(self, handle: zmq.Socket) -> bool:
    """
        Remove a queue from background polling. Returns ``True`` if the queue
        was being monitored, ``False`` otherwise.
        """
    try:
        try:
            self._poller.unregister(handle)
        finally:
            self._queue_callbacks.pop(handle, None)
    except KeyError:
        return False
    return True