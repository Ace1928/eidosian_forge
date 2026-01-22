from __future__ import annotations
import os
import abc
import sys
import anyio
import inspect
import asyncio
import functools
import subprocess
import contextvars
import anyio.from_thread
from concurrent import futures
from anyio._core._eventloop import threadlocals
from lazyops.libs.proxyobj import ProxyObject
from typing import Callable, Coroutine, Any, Union, List, Set, Tuple, TypeVar, Optional, Generator, Awaitable, Iterable, AsyncGenerator, Dict
def ensure_coro_function(self, func: Callable[..., RT]) -> Callable[..., Awaitable[RT]]:
    """
        Ensure that the function is a coroutine
        - Adds contextvars
        """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(executor=self.pool, func=lambda: ctx.run(func, *args, **kwargs))
    return inner