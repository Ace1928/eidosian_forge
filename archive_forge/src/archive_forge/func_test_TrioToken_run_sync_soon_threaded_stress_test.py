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
def test_TrioToken_run_sync_soon_threaded_stress_test() -> None:
    cb_counter = 0

    def cb() -> None:
        nonlocal cb_counter
        cb_counter += 1

    def stress_thread(token: _core.TrioToken) -> None:
        try:
            while True:
                token.run_sync_soon(cb)
                time.sleep(0)
        except _core.RunFinishedError:
            pass

    async def main() -> None:
        token = _core.current_trio_token()
        thread = threading.Thread(target=stress_thread, args=(token,))
        thread.start()
        for _ in range(10):
            start_value = cb_counter
            while cb_counter == start_value:
                await sleep(0.01)
    _core.run(main)
    print(cb_counter)