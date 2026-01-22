import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def user_generator(gendesc, libs):
    """
    A wrapper inserting code calling Numba-compiled *gendesc*.
    """

    def imp(context, builder, sig, args):
        func = context.declare_function(builder.module, gendesc)
        status, retval = context.call_conv.call_function(builder, func, gendesc.restype, gendesc.argtypes, args)
        return (status, retval)
    imp.libs = tuple(libs)
    return imp