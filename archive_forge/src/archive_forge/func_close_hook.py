from __future__ import annotations
import errno
from functools import partial
from typing import TYPE_CHECKING, Awaitable, Callable, NoReturn
import attrs
import trio
from trio import Nursery, StapledStream, TaskStatus
from trio.testing import (
def close_hook() -> None:
    assert trio.current_effective_deadline() == float('-inf')
    record.append('closed')