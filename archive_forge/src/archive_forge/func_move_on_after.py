from __future__ import annotations
import math
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from ..abc._tasks import TaskGroup, TaskStatus
from ._eventloop import get_async_backend
def move_on_after(delay: float | None, shield: bool=False) -> CancelScope:
    """
    Create a cancel scope with a deadline that expires after the given delay.

    :param delay: maximum allowed time (in seconds) before exiting the context block, or
        ``None`` to disable the timeout
    :param shield: ``True`` to shield the cancel scope from external cancellation
    :return: a cancel scope

    """
    deadline = get_async_backend().current_time() + delay if delay is not None else math.inf
    return get_async_backend().create_cancel_scope(deadline=deadline, shield=shield)