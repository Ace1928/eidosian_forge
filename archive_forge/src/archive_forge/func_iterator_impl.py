import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def iterator_impl(iterable_type, iterator_type):
    """
    Decorator a given class as implementing *iterator_type*
    (by providing an `iternext()` method).
    """

    def wrapper(cls):
        iternext = cls.iternext

        @iternext_impl(RefType.BORROWED)
        def iternext_wrapper(context, builder, sig, args, result):
            value, = args
            iterobj = cls(context, builder, value)
            return iternext(iterobj, context, builder, result)
        lower_builtin('iternext', iterator_type)(iternext_wrapper)
        return cls
    return wrapper