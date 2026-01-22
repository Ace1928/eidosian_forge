from __future__ import annotations
from collections import deque
from typing import Callable, Optional, Set
from zope.interface import implementer
from . import IWorker
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
@worker.do
def doWork() -> None:
    try:
        task()
    except BaseException:
        self._logException()

    @self._coordinator.do
    def idleAndPending() -> None:
        self._busyCount -= 1
        self._recycleWorker(not_none_worker)