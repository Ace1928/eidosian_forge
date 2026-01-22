from __future__ import annotations
import asyncio
import contextvars
import sys
import time
from asyncio import get_running_loop
from types import TracebackType
from typing import Any, Awaitable, Callable, TypeVar, cast
def get_traceback_from_context(context: dict[str, Any]) -> TracebackType | None:
    """
    Get the traceback object from the context.
    """
    exception = context.get('exception')
    if exception:
        if hasattr(exception, '__traceback__'):
            return cast(TracebackType, exception.__traceback__)
        else:
            return sys.exc_info()[2]
    return None