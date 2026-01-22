import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def is_not_scalar_zero(builder, value):
    """
    Return a predicate representing whether a *value* is not equal to zero.
    (not exactly "not is_scalar_zero" because of nans)
    """
    return _scalar_pred_against_zero(builder, value, functools.partial(builder.fcmp_unordered, '!='), '!=')