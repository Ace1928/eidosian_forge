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
def for_dict(cls, m):
    """
        A special function that takes a dictionary of promises
        and turns them into a promise for a dictionary of values.
        In other words, this turns an dictionary of promises for values
        into a promise for a dictionary of values.
        """
    dict_type = type(m)
    if not m:
        return cls.resolve(dict_type())

    def handle_success(resolved_values):
        return dict_type(zip(m.keys(), resolved_values))
    return cls.all(m.values()).then(handle_success)