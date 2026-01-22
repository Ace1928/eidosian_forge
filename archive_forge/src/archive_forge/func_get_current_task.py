from __future__ import annotations
from collections.abc import Awaitable, Generator
from typing import Any
from ._eventloop import get_async_backend
def get_current_task() -> TaskInfo:
    """
    Return the current task.

    :return: a representation of the current task

    """
    return get_async_backend().get_current_task()