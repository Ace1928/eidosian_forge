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
def watch_queue(self, queue: zmq.Socket, callback: Callable[[], typing.Any], flags: int=zmq.POLLIN) -> zmq.Socket:
    """
        Call *callback* when zmq *queue* has something to read (when *flags* is
        set to ``POLLIN``, the default) or is available to write (when *flags*
        is set to ``POLLOUT``). No parameters are passed to the callback.
        Returns a handle that may be passed to :meth:`remove_watch_queue`.

        :param queue:
            The zmq queue to poll.

        :param callback:
            The function to call when the poll is successful.

        :param int flags:
            The condition to monitor on the queue (defaults to ``POLLIN``).
        """
    if queue in self._queue_callbacks:
        raise ValueError(f'already watching {queue!r}')
    self._poller.register(queue, flags)
    self._queue_callbacks[queue] = callback
    return queue