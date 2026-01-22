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
@overload(np.random.chisquare)
def chisquare_impl(df):
    if isinstance(df, (types.Float, types.Integer)):

        def _impl(df):
            return 2.0 * np.random.standard_gamma(df / 2.0)
        return _impl