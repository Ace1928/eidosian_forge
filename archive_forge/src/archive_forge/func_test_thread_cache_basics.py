from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
def test_thread_cache_basics() -> None:
    q: Queue[Outcome[object]] = Queue()

    def fn() -> NoReturn:
        raise RuntimeError('hi')

    def deliver(outcome: Outcome[object]) -> None:
        q.put(outcome)
    start_thread_soon(fn, deliver)
    outcome = q.get()
    with pytest.raises(RuntimeError, match='hi'):
        outcome.unwrap()