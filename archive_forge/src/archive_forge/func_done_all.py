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
def done_all(self, handlers=None):
    """
        :type handlers: list[(Any) -> object] | list[((Any) -> object, (Any) -> object)]
        """
    if not handlers:
        return
    for handler in handlers:
        if isinstance(handler, tuple):
            s, f = handler
            self.done(s, f)
        elif isinstance(handler, dict):
            s = handler.get('success')
            f = handler.get('failure')
            self.done(s, f)
        else:
            self.done(handler)