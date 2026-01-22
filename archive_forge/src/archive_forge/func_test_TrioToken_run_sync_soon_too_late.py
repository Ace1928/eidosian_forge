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
def test_TrioToken_run_sync_soon_too_late() -> None:
    token: _core.TrioToken | None = None

    async def main() -> None:
        nonlocal token
        token = _core.current_trio_token()
    _core.run(main)
    with pytest.raises(_core.RunFinishedError):
        not_none(token).run_sync_soon(lambda: None)