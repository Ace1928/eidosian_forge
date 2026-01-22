import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@overload(cmath.tanh)
def impl_cmath_tanh(z):
    if not isinstance(z, types.Complex):
        return

    def tanh_impl(z):
        """cmath.tanh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            real = math.copysign(1.0, x)
            if math.isinf(y):
                imag = 0.0
            else:
                imag = math.copysign(0.0, math.sin(2.0 * y))
            return complex(real, imag)
        tx = math.tanh(x)
        ty = math.tan(y)
        cx = 1.0 / math.cosh(x)
        txty = tx * ty
        denom = 1.0 + txty * txty
        return complex(tx * (1.0 + ty * ty) / denom, ty / denom * cx * cx)
    return tanh_impl