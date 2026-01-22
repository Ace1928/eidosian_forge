import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def increment_index(builder, val):
    """
    Increment an index *val*.
    """
    one = val.type(1)
    return builder.add(val, one, flags=['nsw'])