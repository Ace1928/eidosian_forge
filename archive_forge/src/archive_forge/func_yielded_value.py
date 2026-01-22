import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def yielded_value(self):
    """
        Return the iterator's yielded value, if any.
        """
    return self._pairobj.first