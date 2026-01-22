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
def multinomial_impl(n, pvals, size=None):
    """
            multinomial(..., size=tuple)
            """
    out = np.zeros(size + (len(pvals),), dtype)
    multinomial_inner(n, pvals, out)
    return out