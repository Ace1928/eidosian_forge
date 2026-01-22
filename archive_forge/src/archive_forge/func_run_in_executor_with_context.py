from __future__ import annotations
import asyncio
import contextvars
import sys
import time
from asyncio import get_running_loop
from types import TracebackType
from typing import Any, Awaitable, Callable, TypeVar, cast
def run_in_executor_with_context(func: Callable[..., _T], *args: Any, loop: asyncio.AbstractEventLoop | None=None) -> Awaitable[_T]:
    """
    Run a function in an executor, but make sure it uses the same contextvars.
    This is required so that the function will see the right application.

    See also: https://bugs.python.org/issue34014
    """
    loop = loop or get_running_loop()
    ctx: contextvars.Context = contextvars.copy_context()
    return loop.run_in_executor(None, ctx.run, func, *args)