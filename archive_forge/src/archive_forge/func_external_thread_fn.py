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
def external_thread_fn() -> None:
    try:
        print('running')
        from_thread_run_sync(trio_thread_fn, trio_token=token)
    except KeyboardInterrupt:
        print('ok1')
        record.add('ok1')
    try:
        from_thread_run(trio_thread_afn, trio_token=token)
    except KeyboardInterrupt:
        print('ok2')
        record.add('ok2')