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
def do_abandoned_guest_run() -> None:

    async def abandoned_main(in_host: InHost) -> None:
        in_host(lambda: 1 / 0)
        while True:
            await trio.sleep(0)
    with pytest.raises(ZeroDivisionError):
        trivial_guest_run(abandoned_main)