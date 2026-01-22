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
def test_contextvar_support() -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar('test')
    var.set('before')
    assert var.get() == 'before'

    async def inner() -> None:
        task = _core.current_task()
        assert task.context.get(var) == 'before'
        assert var.get() == 'before'
        var.set('after')
        assert var.get() == 'after'
        assert var in task.context
        assert task.context.get(var) == 'after'
    _core.run(inner)
    assert var.get() == 'before'