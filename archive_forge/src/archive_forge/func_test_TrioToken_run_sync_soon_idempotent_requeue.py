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
def test_TrioToken_run_sync_soon_idempotent_requeue() -> None:
    record: list[None] = []

    def redo(token: _core.TrioToken) -> None:
        record.append(None)
        with suppress(_core.RunFinishedError):
            token.run_sync_soon(redo, token, idempotent=True)

    async def main() -> None:
        token = _core.current_trio_token()
        token.run_sync_soon(redo, token, idempotent=True)
        await _core.checkpoint()
        await _core.checkpoint()
        await _core.checkpoint()
    _core.run(main)
    assert len(record) >= 2