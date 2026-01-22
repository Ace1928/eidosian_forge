import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def yield_(self, value):
    """
        Mark the iterator as yielding the given *value* (a LLVM inst).
        """
    self._pairobj.first = value