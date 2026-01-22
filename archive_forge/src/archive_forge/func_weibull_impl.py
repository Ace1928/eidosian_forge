import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
@overload(np.random.weibull)
def weibull_impl(a):
    if isinstance(a, (types.Float, types.Integer)):

        def _impl(a):
            u = 1.0 - np.random.random()
            return (-math.log(u)) ** (1.0 / a)
        return _impl