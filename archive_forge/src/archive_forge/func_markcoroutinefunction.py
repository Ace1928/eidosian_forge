import asyncio
import asyncio.coroutines
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
from .current_thread_executor import CurrentThreadExecutor
from .local import Local
def markcoroutinefunction(func: _F) -> _F:
    func._is_coroutine = asyncio.coroutines._is_coroutine
    return func