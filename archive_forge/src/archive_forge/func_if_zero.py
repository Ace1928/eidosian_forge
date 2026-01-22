import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def if_zero(builder, value, likely=False):
    """
    Execute the given block if the scalar value is zero.
    """
    with builder.if_then(is_scalar_zero(builder, value), likely=likely):
        yield