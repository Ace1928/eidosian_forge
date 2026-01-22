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
@overload(np.random.standard_gamma)
@overload(np.random.gamma)
def ol_np_random_gamma1(shape):
    if isinstance(shape, (types.Float, types.Integer)):
        return lambda shape: np.random.gamma(shape, 1.0)