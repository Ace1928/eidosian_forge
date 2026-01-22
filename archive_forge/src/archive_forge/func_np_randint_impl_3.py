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
@overload(np.random.randint)
def np_randint_impl_3(low, high, size):
    if isinstance(low, types.Integer) and isinstance(high, types.Integer) and is_nonelike(size):
        return lambda low, high, size: np.random.randint(low, high)
    if isinstance(low, types.Integer) and isinstance(high, types.Integer) and (isinstance(size, types.Integer) or (isinstance(size, types.UniTuple) and isinstance(size.dtype, types.Integer))):
        bitwidth = max(low.bitwidth, high.bitwidth)
        result_type = getattr(np, f'int{bitwidth}')

        def _impl(low, high, size):
            out = np.empty(size, dtype=result_type)
            out_flat = out.flat
            for idx in range(out.size):
                out_flat[idx] = np.random.randint(low, high)
            return out
        return _impl