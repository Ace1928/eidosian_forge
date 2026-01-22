import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def finish_at_shutdown(self, handle_timeouts=False):
    self._shutdown_complete = True
    get = self.get
    outqueue = self.outqueue
    cache = self.cache
    poll = self.poll
    join_exited_workers = self.join_exited_workers
    check_timeouts = self.check_timeouts
    on_state_change = self.on_state_change
    time_terminate = None
    while cache and self._state != TERMINATE:
        if check_timeouts is not None:
            check_timeouts()
        try:
            ready, task = poll(1.0)
        except (IOError, EOFError) as exc:
            debug('result handler got %r -- exiting', exc)
            return
        if ready:
            if task is None:
                debug('result handler ignoring extra sentinel')
                continue
            on_state_change(task)
        try:
            join_exited_workers(shutdown=True)
        except WorkersJoined:
            now = monotonic()
            if not time_terminate:
                time_terminate = now
            else:
                if now - time_terminate > 5.0:
                    debug('result handler exiting: timed out')
                    break
                debug('result handler: all workers terminated, timeout in %ss', abs(min(now - time_terminate - 5.0, 0)))
    if hasattr(outqueue, '_reader'):
        debug('ensuring that outqueue is not full')
        try:
            for i in range(10):
                if not outqueue._reader.poll():
                    break
                get()
        except (IOError, EOFError):
            pass
    debug('result handler exiting: len(cache)=%s, thread._state=%s', len(cache), self._state)