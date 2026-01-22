from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def expensive(fn: Callable[[TSelf], TRetVal]) -> Callable[[TSelf], TRetVal]:
    """A decorator to indicate that a method shouldn't be called more than once.

    Normally, this does nothing.  During testing, this raises an exception if
    called more than once.

    """
    if env.TESTING:
        attr = '_once_' + fn.__name__

        def _wrapper(self: TSelf) -> TRetVal:
            if hasattr(self, attr):
                raise AssertionError(f"Shouldn't have called {fn.__name__} more than once")
            setattr(self, attr, True)
            return fn(self)
        return _wrapper
    else:
        return fn