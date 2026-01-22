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
@overload(random.randint)
def randint_impl_1(a, b):
    if isinstance(a, types.Integer) and isinstance(b, types.Integer):
        return lambda a, b: random.randrange(a, b + 1, 1)