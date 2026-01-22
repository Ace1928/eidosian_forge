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
def test_calling_asyncio_function_gives_nice_error() -> None:

    async def child_xyzzy() -> None:
        await create_asyncio_future_in_new_loop()

    async def misguided() -> None:
        await child_xyzzy()
    with pytest.raises(TypeError, match='asyncio') as excinfo:
        _core.run(misguided)
    assert any((entry.name == 'child_xyzzy' for entry in excinfo.traceback))