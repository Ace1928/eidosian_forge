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
def pop_deadline(self) -> None:
    entry = self._deadlines.pop(-1)
    entry.valid = False