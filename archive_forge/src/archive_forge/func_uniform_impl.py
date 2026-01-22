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
def uniform_impl(state, a_preprocessor, b_preprocessor):

    def impl(context, builder, sig, args):
        state_ptr = get_state_ptr(context, builder, state)
        a, b = args
        a = a_preprocessor(builder, a)
        b = b_preprocessor(builder, b)
        width = builder.fsub(b, a)
        r = get_next_double(context, builder, state_ptr)
        return builder.fadd(a, builder.fmul(width, r))
    return impl