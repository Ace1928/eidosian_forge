from collections import namedtuple
from functools import partial, wraps
from sys import version_info, exc_info
from threading import RLock
from types import TracebackType
from weakref import WeakKeyDictionary
from .async_ import Async
from .compat import (
from .utils import deprecated, integer_types, string_types, text_type, binary_type, warn
from .promise_list import PromiseList
from .schedulers.immediate import ImmediateScheduler
from typing import TypeVar, Generic
@classmethod
def safe(cls, fn):
    from functools import wraps
    if not cls._safe_resolved_promise:
        cls._safe_resolved_promise = Promise.resolve(None)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return cls._safe_resolved_promise.then(lambda v: fn(*args, **kwargs))
    return wrapper