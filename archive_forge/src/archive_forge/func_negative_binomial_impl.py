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
@overload(np.random.negative_binomial)
def negative_binomial_impl(n, p):
    if isinstance(n, types.Integer) and isinstance(p, (types.Float, types.Integer)):

        def _impl(n, p):
            if n <= 0:
                raise ValueError('negative_binomial(): n <= 0')
            if p < 0.0 or p > 1.0:
                raise ValueError('negative_binomial(): p outside of [0, 1]')
            Y = np.random.gamma(n, (1.0 - p) / p)
            return np.random.poisson(Y)
        return _impl