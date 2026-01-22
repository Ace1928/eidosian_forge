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
def background_task(self, func: Callable[..., RT], *args, task_callback: Optional[Callable]=None, task_callback_args: Optional[Tuple]=None, task_callback_kwargs: Optional[Dict]=None, **kwargs) -> Awaitable[RT]:
    """
        Creates a background task
        """
    if inspect.isawaitable(func):
        task = asyncio.create_task(func)
    else:
        task = asyncio.create_task(self.asyncish(func, *args, **kwargs))
    self.add_task(task, task_callback, callback_args=task_callback_args, callback_kwargs=task_callback_kwargs)
    return task