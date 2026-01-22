from __future__ import annotations
import contextlib
import inspect
import signal
import threading
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterator
import outcome
import pytest
from trio.testing import RaisesGroup
from ... import _core
from ..._abc import Instrument
from ..._timeouts import sleep
from ..._util import signal_raise
from ...testing import wait_all_tasks_blocked
def test_ki_with_broken_threads() -> None:
    thread = threading.main_thread()
    original = threading._active[thread.ident]
    try:
        del threading._active[thread.ident]

        @_core.enable_ki_protection
        async def inner() -> None:
            assert signal.getsignal(signal.SIGINT) != signal.default_int_handler
        _core.run(inner)
    finally:
        threading._active[thread.ident] = original