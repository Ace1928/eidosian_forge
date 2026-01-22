from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def async_wraps(cls: type[object], wrapped_cls: type[object], attr_name: str) -> Callable[[CallT], CallT]:
    """Similar to wraps, but for async wrappers of non-async functions."""

    def decorator(func: CallT) -> CallT:
        func.__name__ = attr_name
        func.__qualname__ = '.'.join((cls.__qualname__, attr_name))
        func.__doc__ = f'Like :meth:`~{wrapped_cls.__module__}.{wrapped_cls.__qualname__}.{attr_name}`, but async.'
        return func
    return decorator