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
def test_nice_error_on_bad_calls_to_run_or_spawn() -> None:

    def bad_call_run(func: Callable[..., Awaitable[object]], *args: tuple[object, ...]) -> None:
        _core.run(func, *args)

    def bad_call_spawn(func: Callable[..., Awaitable[object]], *args: tuple[object, ...]) -> None:

        async def main() -> None:
            async with _core.open_nursery() as nursery:
                nursery.start_soon(func, *args)
        _core.run(main)

    async def f() -> None:
        pass

    async def async_gen(arg: T) -> AsyncGenerator[T, None]:
        yield arg
    with pytest.raises(TypeError, match='^Trio was expecting an async function, but instead it got a coroutine object <.*>'):
        bad_call_run(f())
    with pytest.raises(TypeError, match='expected an async function but got an async generator'):
        bad_call_run(async_gen, 0)
    with RaisesGroup(Matcher(TypeError, 'expecting an async function')):
        bad_call_spawn(f())
    with RaisesGroup(Matcher(TypeError, 'expected an async function but got an async generator')):
        bad_call_spawn(async_gen, 0)