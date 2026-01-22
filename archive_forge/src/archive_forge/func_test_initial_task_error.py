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
def test_initial_task_error() -> None:

    async def main(x: object) -> NoReturn:
        raise ValueError(x)
    with pytest.raises(ValueError, match='^17$') as excinfo:
        _core.run(main, 17)
    assert excinfo.value.args == (17,)