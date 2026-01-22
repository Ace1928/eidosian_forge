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
def test_ki_is_good_neighbor() -> None:
    try:
        orig = signal.getsignal(signal.SIGINT)

        def my_handler(signum: object, frame: object) -> None:
            pass

        async def main() -> None:
            signal.signal(signal.SIGINT, my_handler)
        _core.run(main)
        assert signal.getsignal(signal.SIGINT) is my_handler
    finally:
        signal.signal(signal.SIGINT, orig)