import datetime
import time
from collections import deque
from contextlib import contextmanager
from weakref import proxy
from dateutil.parser import isoparse
from kombu.utils.objects import cached_property
from vine import Thenable, barrier, promise
from . import current_app, states
from ._state import _set_task_join_will_block, task_join_will_block
from .app import app_or_default
from .exceptions import ImproperlyConfigured, IncompleteStream, TimeoutError
from .utils.graph import DependencyGraph, GraphFormatter
def completed_count(self):
    """Task completion count.

        Note that `complete` means `successful` in this context. In other words, the
        return value of this method is the number of ``successful`` tasks.

        Returns:
            int: the number of complete (i.e. successful) tasks.
        """
    return sum((int(result.successful()) for result in self.results))