import bisect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import islice
from operator import itemgetter
from time import time
from typing import Mapping, Optional  # noqa
from weakref import WeakSet, ref
from kombu.clocks import timetuple
from kombu.utils.objects import cached_property
from celery import states
from celery.utils.functional import LRUCache, memoize, pass1
from celery.utils.log import get_logger
def tasks_by_time(self, limit=None, reverse: bool=True):
    """Generator yielding tasks ordered by time.

        Yields:
            Tuples of ``(uuid, Task)``.
        """
    _heap = self._taskheap
    if reverse:
        _heap = reversed(_heap)
    seen = set()
    for evtup in islice(_heap, 0, limit):
        task = evtup[3]()
        if task is not None:
            uuid = task.uuid
            if uuid not in seen:
                yield (uuid, task)
                seen.add(uuid)