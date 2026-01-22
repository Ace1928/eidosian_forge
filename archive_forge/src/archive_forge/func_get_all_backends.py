from __future__ import annotations
import math
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar
import sniffio
def get_all_backends() -> tuple[str, ...]:
    """Return a tuple of the names of all built-in backends."""
    return BACKENDS