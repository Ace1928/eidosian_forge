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
def got_signal(proc: Process, sig: SignalType) -> bool:
    if not TYPE_CHECKING and posix or sys.platform != 'win32':
        return proc.returncode == -sig
    else:
        return proc.returncode != 0