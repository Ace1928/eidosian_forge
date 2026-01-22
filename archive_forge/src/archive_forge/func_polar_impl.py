import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@overload(cmath.polar)
def polar_impl(x):
    if not isinstance(x, types.Complex):
        return

    def impl(x):
        r, i = (x.real, x.imag)
        return (math.hypot(r, i), math.atan2(i, r))
    return impl