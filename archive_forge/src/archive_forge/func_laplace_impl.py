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
def laplace_impl(loc, scale):
    U = np.random.random()
    if U < 0.5:
        return loc + scale * math.log(U + U)
    else:
        return loc - scale * math.log(2.0 - U - U)