import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def numba_typeref_ctor(*args, **kwargs):
    """A stub for use internally by Numba when a call is emitted
    on a TypeRef.
    """
    raise NotImplementedError('This function should not be executed.')