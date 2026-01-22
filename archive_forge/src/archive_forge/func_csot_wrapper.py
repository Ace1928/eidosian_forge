from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
@functools.wraps(func)
def csot_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
    if get_timeout() is None:
        timeout = self._timeout
        if timeout is not None:
            with _TimeoutContext(timeout):
                return func(self, *args, **kwargs)
    return func(self, *args, **kwargs)