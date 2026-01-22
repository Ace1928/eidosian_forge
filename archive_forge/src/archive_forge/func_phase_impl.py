import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@overload(cmath.phase)
def phase_impl(x):
    """cmath.phase(x + y j)"""
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        return math.atan2(x.imag, x.real)
    return impl