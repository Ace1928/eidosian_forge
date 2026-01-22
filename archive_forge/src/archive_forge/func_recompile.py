import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def recompile(self):
    """
        Recompile all signatures afresh.
        """
    sigs = list(self.overloads)
    old_can_compile = self._can_compile
    self._make_finalizer()()
    self._reset_overloads()
    self._cache.flush()
    self._can_compile = True
    try:
        for sig in sigs:
            self.compile(sig)
    finally:
        self._can_compile = old_can_compile