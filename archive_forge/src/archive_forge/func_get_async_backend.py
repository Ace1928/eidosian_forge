from __future__ import annotations
import math
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar
import sniffio
def get_async_backend(asynclib_name: str | None=None) -> AsyncBackend:
    if asynclib_name is None:
        asynclib_name = sniffio.current_async_library()
    modulename = 'anyio._backends._' + asynclib_name
    try:
        module = sys.modules[modulename]
    except KeyError:
        module = import_module(modulename)
    return getattr(module, 'backend_class')