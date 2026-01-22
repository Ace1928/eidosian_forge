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
@restore_unraisablehook()
def test_broken_abort() -> None:

    async def main() -> None:
        await _core.checkpoint()
        await _core.checkpoint()
        with _core.CancelScope() as scope:
            scope.cancel()
            await _core.wait_task_rescheduled(lambda _: None)
    with pytest.raises(_core.TrioInternalError):
        _core.run(main)
    gc_collect_harder()