from __future__ import annotations
import asyncio
import contextlib
import contextvars
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from functools import partial
from math import inf
from typing import (
import pytest
from outcome import Outcome
import trio
import trio.testing
from trio.abc import Instrument
from ..._util import signal_raise
from .tutil import gc_collect_harder, restore_unraisablehook
def test_guest_mode_autojump_clock_threshold_changing() -> None:
    clock = trio.testing.MockClock()
    DURATION = 120

    async def trio_main(in_host: InHost) -> None:
        assert trio.current_time() == 0
        in_host(lambda: setattr(clock, 'autojump_threshold', 0))
        await trio.sleep(DURATION)
        assert trio.current_time() == DURATION
    start = time.monotonic()
    trivial_guest_run(trio_main, clock=clock)
    end = time.monotonic()
    assert end - start < DURATION / 2