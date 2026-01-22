from __future__ import annotations
import os
from collections import deque
from queue import Empty
from queue import LifoQueue as _LifoQueue
from typing import TYPE_CHECKING
from . import exceptions
from .utils.compat import register_after_fork
from .utils.functional import lazy
def force_close_all(self):
    """Close and remove all resources in the pool (also those in use).

        Used to close resources from parent processes after fork
        (e.g. sockets/connections).
        """
    if self._closed:
        return
    self._closed = True
    dirty = self._dirty
    resource = self._resource
    while 1:
        try:
            dres = dirty.pop()
        except KeyError:
            break
        try:
            self.collect_resource(dres)
        except AttributeError:
            pass
    while 1:
        try:
            res = resource.queue.pop()
        except IndexError:
            break
        try:
            self.collect_resource(res)
        except AttributeError:
            pass