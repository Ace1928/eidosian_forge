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
@overload(random.lognormvariate)
def lognormvariate_impl(mu, sigma):
    if isinstance(mu, types.Float) and isinstance(sigma, types.Float):
        fn = register_jitable(_lognormvariate_impl(random.gauss))
        return lambda mu, sigma: fn(mu, sigma)