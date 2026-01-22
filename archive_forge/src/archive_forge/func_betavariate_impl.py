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
@overload(random.betavariate)
def betavariate_impl(alpha, beta):
    if isinstance(alpha, (types.Float, types.Integer)) and isinstance(beta, (types.Float, types.Integer)):
        return _betavariate_impl(random.gammavariate)