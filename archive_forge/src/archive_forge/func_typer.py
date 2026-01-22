from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
def typer(coef, domain=None, window=None):
    default_domain = types.Array(types.int64, 1, 'C')
    double_domain = types.Array(types.double, 1, 'C')
    default_window = types.Array(types.int64, 1, 'C')
    double_window = types.Array(types.double, 1, 'C')
    double_coef = types.Array(types.double, 1, 'C')
    warnings.warn('Polynomial class is experimental', category=NumbaExperimentalFeatureWarning)
    if isinstance(coef, types.Array) and all([a is None for a in (domain, window)]):
        if coef.ndim == 1:
            return types.PolynomialType(double_coef, default_domain, default_window, 1)
        else:
            msg = 'Coefficient array is not 1-d'
            raise NumbaValueError(msg)
    elif all([isinstance(a, types.Array) for a in (coef, domain, window)]):
        if coef.ndim == 1:
            if all([a.ndim == 1 for a in (domain, window)]):
                return types.PolynomialType(double_coef, double_domain, double_window, 3)
        else:
            msg = 'Coefficient array is not 1-d'
            raise NumbaValueError(msg)