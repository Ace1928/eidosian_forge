import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def log10_impl(z):
    """cmath.log10(z)"""
    z = cmath.log(z)
    return complex(z.real / LN_10, z.imag / LN_10)