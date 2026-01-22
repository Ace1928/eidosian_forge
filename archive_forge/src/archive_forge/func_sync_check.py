from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def sync_check() -> None:
    from_thread_run_sync(cancel_scope.cancel)
    try:
        from_thread_run_sync(bool)
    except _core.Cancelled:
        queue.put(True)
    else:
        queue.put(False)