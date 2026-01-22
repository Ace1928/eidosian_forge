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
def push_deadline(self, deadline: float, timeout_error: TimeoutError) -> None:
    if self._deadlines:
        entry = self._deadlines[-1]
        if entry.deadline < deadline:
            deadline = entry.deadline
            timeout_error = entry.timeout_error
    entry = DeadlineEntry(self, deadline, timeout_error)
    self.scheduler.add_deadline(entry)
    self._deadlines.append(entry)