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
def test_TrioToken_identity() -> None:

    async def get_and_check_token() -> _core.TrioToken:
        token = _core.current_trio_token()
        assert token is _core.current_trio_token()
        return token
    t1 = _core.run(get_and_check_token)
    t2 = _core.run(get_and_check_token)
    assert t1 is not t2
    assert t1 != t2
    assert hash(t1) != hash(t2)