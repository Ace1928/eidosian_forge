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
@property
def in_async_loop(self) -> bool:
    """
        Returns True if the current thread is in an async loop
        """
    try:
        return asyncio.get_running_loop() is not None
    except RuntimeError:
        return False