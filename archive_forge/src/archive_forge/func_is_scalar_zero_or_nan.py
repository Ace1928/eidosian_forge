import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def is_scalar_zero_or_nan(builder, value):
    """
    Return a predicate representing whether *value* is equal to either zero
    or NaN.
    """
    return _scalar_pred_against_zero(builder, value, functools.partial(builder.fcmp_unordered, '=='), '==')