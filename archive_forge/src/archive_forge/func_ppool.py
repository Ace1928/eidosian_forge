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
def ppool(self) -> futures.ProcessPoolExecutor:
    """
        Returns the ProcessPoolExecutor
        """
    if self._ppool is None:
        self._ppool = futures.ProcessPoolExecutor(max_workers=self.max_workers)
    return self._ppool