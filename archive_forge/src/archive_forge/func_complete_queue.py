import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
def complete_queue(self):
    if not self._enqueue:
        return
    with self._confirmation_lock:
        self._queue.put(True)
        self._confirmation_event.wait()
        self._confirmation_event.clear()