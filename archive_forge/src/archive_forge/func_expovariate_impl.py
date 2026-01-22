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
@overload(random.expovariate)
def expovariate_impl(lambd):
    if isinstance(lambd, types.Float):

        def _impl(lambd):
            """Exponential distribution.  Taken from CPython.
            """
            return -math.log(1.0 - random.random()) / lambd
        return _impl