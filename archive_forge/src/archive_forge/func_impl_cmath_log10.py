import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@overload(cmath.log10)
def impl_cmath_log10(z):
    if not isinstance(z, types.Complex):
        return
    LN_10 = 2.302585092994046

    def log10_impl(z):
        """cmath.log10(z)"""
        z = cmath.log(z)
        return complex(z.real / LN_10, z.imag / LN_10)
    return log10_impl