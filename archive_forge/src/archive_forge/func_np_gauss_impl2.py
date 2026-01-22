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
@overload(np.random.normal)
def np_gauss_impl2(loc, scale):
    if isinstance(loc, (types.Float, types.Integer)) and isinstance(scale, (types.Float, types.Integer)):

        @intrinsic
        def _impl(typingcontext, loc, scale):
            loc_preprocessor = _double_preprocessor(loc)
            scale_preprocessor = _double_preprocessor(scale)
            return (signature(types.float64, loc, scale), _gauss_impl('np', loc_preprocessor, scale_preprocessor))
        return lambda loc, scale: _impl(loc, scale)