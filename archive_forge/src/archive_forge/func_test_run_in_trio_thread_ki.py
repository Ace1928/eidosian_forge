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
def test_run_in_trio_thread_ki() -> None:
    record = set()

    async def check_run_in_trio_thread() -> None:
        token = _core.current_trio_token()

        def trio_thread_fn() -> None:
            print('in Trio thread')
            assert not _core.currently_ki_protected()
            print('ki_self')
            try:
                ki_self()
            finally:
                import sys
                print('finally', sys.exc_info())

        async def trio_thread_afn() -> None:
            trio_thread_fn()

        def external_thread_fn() -> None:
            try:
                print('running')
                from_thread_run_sync(trio_thread_fn, trio_token=token)
            except KeyboardInterrupt:
                print('ok1')
                record.add('ok1')
            try:
                from_thread_run(trio_thread_afn, trio_token=token)
            except KeyboardInterrupt:
                print('ok2')
                record.add('ok2')
        thread = threading.Thread(target=external_thread_fn)
        thread.start()
        print('waiting')
        while thread.is_alive():
            await sleep(0.01)
        print('waited, joining')
        thread.join()
        print('done')
    _core.run(check_run_in_trio_thread)
    assert record == {'ok1', 'ok2'}