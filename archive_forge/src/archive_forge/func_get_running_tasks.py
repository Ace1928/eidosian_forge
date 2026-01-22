from __future__ import annotations
from collections.abc import Awaitable, Generator
from typing import Any
from ._eventloop import get_async_backend
def get_running_tasks() -> list[TaskInfo]:
    """
    Return a list of running tasks in the current event loop.

    :return: a list of task info objects

    """
    return get_async_backend().get_running_tasks()