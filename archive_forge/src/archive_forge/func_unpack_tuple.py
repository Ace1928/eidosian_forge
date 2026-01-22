import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def unpack_tuple(builder, tup, count=None):
    """
    Unpack an array or structure of values, return a Python tuple.
    """
    if count is None:
        count = len(tup.type.elements)
    vals = [builder.extract_value(tup, i) for i in range(count)]
    return vals