from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(np.polynomial.polynomial.Polynomial)
def typeof_numpy_polynomial(val, c):
    coef = typeof(val.coef)
    domain = typeof(val.domain)
    window = typeof(val.window)
    return types.PolynomialType(coef, domain, window)