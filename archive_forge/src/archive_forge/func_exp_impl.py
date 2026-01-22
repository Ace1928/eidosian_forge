import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@lower(cmath.exp, types.Complex)
@intrinsic_complex_unary
def exp_impl(x, y, x_is_finite, y_is_finite):
    """cmath.exp(x + y j)"""
    if x_is_finite:
        if y_is_finite:
            c = math.cos(y)
            s = math.sin(y)
            r = math.exp(x)
            return complex(r * c, r * s)
        else:
            return complex(NAN, NAN)
    elif math.isnan(x):
        if y:
            return complex(x, x)
        else:
            return complex(x, y)
    elif x > 0.0:
        if y_is_finite:
            real = math.cos(y)
            imag = math.sin(y)
            if real != 0:
                real *= x
            if imag != 0:
                imag *= x
            return complex(real, imag)
        else:
            return complex(x, NAN)
    elif y_is_finite:
        r = math.exp(x)
        c = math.cos(y)
        s = math.sin(y)
        return complex(r * c, r * s)
    else:
        r = 0
        return complex(r, r)