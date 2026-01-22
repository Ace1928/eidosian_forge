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
def is_thenable(cls, obj):
    """
        A utility function to determine if the specified
        object is a promise using "duck typing".
        """
    _type = obj.__class__
    if obj is None or _type in BASE_TYPES:
        return False
    return issubclass(_type, Promise) or iscoroutine(obj) or is_future_like(_type)