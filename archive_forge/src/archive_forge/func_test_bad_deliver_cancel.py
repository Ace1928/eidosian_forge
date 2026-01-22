from __future__ import annotations
import gc
import os
import random
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path as SyncPath
from signal import Signals
from typing import (
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import (
from .._core._tests.tutil import skip_if_fbsd_pipes_broken, slow
from ..lowlevel import open_process
from ..testing import MockClock, assert_no_checkpoints, wait_all_tasks_blocked
def test_bad_deliver_cancel() -> None:

    async def custom_deliver_cancel(proc: Process) -> None:
        proc.terminate()
        raise ValueError('foo')

    async def do_stuff() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(partial(run_process, SLEEP(9999), deliver_cancel=custom_deliver_cancel))
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()
    with RaisesGroup(RaisesGroup(Matcher(ValueError, '^foo$'))):
        _core.run(do_stuff, strict_exception_groups=True)