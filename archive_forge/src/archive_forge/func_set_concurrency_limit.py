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
def set_concurrency_limit(limit: Optional[int]=None):
    """
    Set the concurrency limit
    """
    global _concurrency_limit
    if limit is None:
        limit = os.cpu_count() * 4
    _concurrency_limit = limit