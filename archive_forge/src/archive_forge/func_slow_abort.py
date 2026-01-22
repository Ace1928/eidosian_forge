from __future__ import annotations
import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
import outcome
import pytest
import sniffio
from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD
from .tutil import (
def slow_abort(raise_cancel: _core.RaiseCancelT) -> _core.Abort:
    record.append('abort-called')
    result = outcome.capture(raise_cancel)
    token.run_sync_soon(_core.reschedule, task, result)
    return _core.Abort.FAILED