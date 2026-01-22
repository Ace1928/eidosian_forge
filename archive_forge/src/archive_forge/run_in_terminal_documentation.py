from __future__ import annotations
from asyncio import Future, ensure_future
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, TypeVar
from prompt_toolkit.eventloop import run_in_executor_with_context
from .current import get_app_or_none

    Asynchronous context manager that suspends the current application and runs
    the body in the terminal.

    .. code::

        async def f():
            async with in_terminal():
                call_some_function()
                await call_some_async_function()
    