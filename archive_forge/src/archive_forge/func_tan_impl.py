import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def tan_impl(z):
    """cmath.tan(z) = -j * cmath.tanh(z j)"""
    r = cmath.tanh(complex(-z.imag, z.real))
    return complex(r.imag, -r.real)