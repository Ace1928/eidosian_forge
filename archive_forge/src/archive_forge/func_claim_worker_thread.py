from __future__ import annotations
import math
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar
import sniffio
@contextmanager
def claim_worker_thread(backend_class: type[AsyncBackend], token: object) -> Generator[Any, None, None]:
    threadlocals.current_async_backend = backend_class
    threadlocals.current_token = token
    try:
        yield
    finally:
        del threadlocals.current_async_backend
        del threadlocals.current_token