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
@overload(np.random.beta)
def ol_np_random_beta(a, b):
    if isinstance(a, (types.Float, types.Integer)) and isinstance(b, (types.Float, types.Integer)):
        fn = register_jitable(_betavariate_impl(np.random.gamma))

        def impl(a, b):
            return fn(a, b)
        return impl