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
def test_TrioToken_run_sync_soon_late_crash() -> None:
    record: list[str] = []
    saved: list[AsyncGenerator[int, None]] = []

    async def agen() -> AsyncGenerator[int, None]:
        token = _core.current_trio_token()
        try:
            yield 1
        finally:
            token.run_sync_soon(lambda: {}['nope'])
            token.run_sync_soon(lambda: record.append('2nd ran'))

    async def main() -> None:
        saved.append(agen())
        await saved[-1].asend(None)
        record.append('main exiting')
    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)
    assert RaisesGroup(KeyError).matches(excinfo.value.__cause__)
    assert record == ['main exiting', '2nd ran']