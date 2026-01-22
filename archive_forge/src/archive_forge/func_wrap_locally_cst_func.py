from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def wrap_locally_cst_func(func):
    """
    Return a function that returns the same arguments as func, but
    after converting any AffineScalarFunc object to its nominal value.

    This function is useful for wrapping functions that are locally
    constant: the uncertainties should have no role in the result
    (since they are supposed to keep the function linear and hence,
    here, constant).
    """

    def wrapped_func(*args, **kwargs):
        args_float = map(uncert_core.nominal_value, args)
        kwargs_float = dict(((arg_name, uncert_core.nominal_value(value)) for arg_name, value in kwargs.items()))
        return func(*args_float, **kwargs_float)
    return wrapped_func