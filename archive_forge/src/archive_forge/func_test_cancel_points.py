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
def test_cancel_points() -> None:

    async def main1() -> None:
        with _core.CancelScope() as scope:
            await _core.checkpoint_if_cancelled()
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint_if_cancelled()
    _core.run(main1)

    async def main2() -> None:
        with _core.CancelScope() as scope:
            await _core.checkpoint()
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
    _core.run(main2)

    async def main3() -> None:
        with _core.CancelScope() as scope:
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await sleep_forever()
    _core.run(main3)

    async def main4() -> None:
        with _core.CancelScope() as scope:
            scope.cancel()
            await _core.cancel_shielded_checkpoint()
            await _core.cancel_shielded_checkpoint()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
    _core.run(main4)