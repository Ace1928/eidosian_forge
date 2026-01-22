import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def dtype_short_repr(dtype):
    """
    Convert a dtype to a short form which evaluates to the same dtype.

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> dt = np.int64([1, 2]).dtype
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    if type(dtype).__repr__ != np.dtype.__repr__:
        return repr(dtype)
    if dtype.names is not None:
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        return "'%s'" % str(dtype)
    typename = dtype.name
    if not dtype.isnative:
        return "'%s'" % str(dtype)
    if typename and (not (typename[0].isalpha() and typename.isalnum())):
        typename = repr(typename)
    return typename