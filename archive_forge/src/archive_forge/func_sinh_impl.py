import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def sinh_impl(z):
    """cmath.sinh(z)"""
    x = z.real
    y = z.imag
    if math.isinf(x):
        if math.isnan(y):
            real = x
            imag = y
        else:
            real = math.cos(y)
            imag = math.sin(y)
            if real != 0.0:
                real *= x
            if imag != 0.0:
                imag *= abs(x)
        return complex(real, imag)
    return complex(math.cos(y) * math.sinh(x), math.sin(y) * math.cosh(x))