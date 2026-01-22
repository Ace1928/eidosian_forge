from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def test_await_in_trio_thread_while_main_exits() -> None:
    record = []
    ev = Event()

    async def trio_fn() -> None:
        record.append('sleeping')
        ev.set()
        await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)

    def thread_fn(token: _core.TrioToken) -> None:
        try:
            from_thread_run(trio_fn, trio_token=token)
        except _core.Cancelled:
            record.append('cancelled')

    async def main() -> threading.Thread:
        token = _core.current_trio_token()
        thread = threading.Thread(target=thread_fn, args=(token,))
        thread.start()
        await ev.wait()
        assert record == ['sleeping']
        return thread
    thread = _core.run(main)
    thread.join()
    assert record == ['sleeping', 'cancelled']