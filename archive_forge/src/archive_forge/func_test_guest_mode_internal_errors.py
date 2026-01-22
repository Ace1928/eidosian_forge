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
def test_guest_mode_internal_errors(monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder) -> None:
    with monkeypatch.context() as m:

        async def crash_in_run_loop(in_host: InHost) -> None:
            m.setattr('trio._core._run.GLOBAL_RUN_CONTEXT.runner.runq', 'HI')
            await trio.sleep(1)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_run_loop)
    with monkeypatch.context() as m:

        async def crash_in_io(in_host: InHost) -> None:
            m.setattr('trio._core._run.TheIOManager.get_events', None)
            await trio.sleep(0)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_io)
    with monkeypatch.context() as m:

        async def crash_in_worker_thread_io(in_host: InHost) -> None:
            t = threading.current_thread()
            old_get_events = trio._core._run.TheIOManager.get_events

            def bad_get_events(*args: Any) -> object:
                if threading.current_thread() is not t:
                    raise ValueError('oh no!')
                else:
                    return old_get_events(*args)
            m.setattr('trio._core._run.TheIOManager.get_events', bad_get_events)
            await trio.sleep(1)
        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_worker_thread_io)
    gc_collect_harder()