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
def then_all(self, handlers=None):
    """
        Utility function which calls 'then' for each handler provided. Handler can either
        be a function in which case it is used as success handler, or a tuple containing
        the success and the failure handler, where each of them could be None.
        :type handlers: list[(Any) -> object] | list[((Any) -> object, (Any) -> object)]
        :param handlers
        :rtype : list[Promise]
        """
    if not handlers:
        return []
    promises = []
    for handler in handlers:
        if isinstance(handler, tuple):
            s, f = handler
            promises.append(self.then(s, f))
        elif isinstance(handler, dict):
            s = handler.get('success')
            f = handler.get('failure')
            promises.append(self.then(s, f))
        else:
            promises.append(self.then(handler))
    return promises