import enum
import functools
import heapq
import itertools
import signal
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar
from typing import (
import duet.futuretools as futuretools
def get_next_deadline(self) -> Optional[float]:
    while self._deadlines:
        if not self._deadlines[0].valid:
            heapq.heappop(self._deadlines)
            continue
        return self._deadlines[0].deadline
    return None