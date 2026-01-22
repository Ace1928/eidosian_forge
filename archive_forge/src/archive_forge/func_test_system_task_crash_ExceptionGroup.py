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
def test_system_task_crash_ExceptionGroup() -> None:

    async def crasher1() -> NoReturn:
        raise KeyError

    async def crasher2() -> NoReturn:
        raise ValueError

    async def system_task() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher1)
            nursery.start_soon(crasher2)

    async def main() -> None:
        _core.spawn_system_task(system_task)
        await sleep_forever()
    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)
    assert RaisesGroup(RaisesGroup(RaisesGroup(KeyError, ValueError))).matches(excinfo.value.__cause__)