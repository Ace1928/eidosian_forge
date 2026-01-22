from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
def clamp_remaining(max_timeout: float) -> float:
    """Return the remaining timeout clamped to a max value."""
    timeout = remaining()
    if timeout is None:
        return max_timeout
    return min(timeout, max_timeout)