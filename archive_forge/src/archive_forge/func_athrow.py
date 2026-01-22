import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def athrow(self, type, value=None, traceback=None):
    return self._do_it(self._it.throw, type, value, traceback)