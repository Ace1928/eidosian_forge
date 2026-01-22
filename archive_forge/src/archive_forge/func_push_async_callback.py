import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def push_async_callback(self, callback, /, *args, **kwds):
    """Registers an arbitrary coroutine function and arguments.

        Cannot suppress exceptions.
        """
    _exit_wrapper = self._create_async_cb_wrapper(callback, *args, **kwds)
    _exit_wrapper.__wrapped__ = callback
    self._push_exit_callback(_exit_wrapper, False)
    return callback