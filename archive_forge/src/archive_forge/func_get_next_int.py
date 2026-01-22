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
def get_next_int(context, builder, state_ptr, nbits, is_numpy):
    """
    Get the next integer with width *nbits*.
    """
    c32 = ir.Constant(nbits.type, 32)

    def get_shifted_int(nbits):
        shift = builder.sub(c32, nbits)
        y = get_next_int32(context, builder, state_ptr)
        if nbits.type.width < y.type.width:
            shift = builder.zext(shift, y.type)
        elif nbits.type.width > y.type.width:
            shift = builder.trunc(shift, y.type)
        if is_numpy:
            mask = builder.not_(ir.Constant(y.type, 0))
            mask = builder.lshr(mask, shift)
            return builder.and_(y, mask)
        else:
            return builder.lshr(y, shift)
    ret = cgutils.alloca_once_value(builder, ir.Constant(int64_t, 0))
    is_32b = builder.icmp_unsigned('<=', nbits, c32)
    with builder.if_else(is_32b) as (ifsmall, iflarge):
        with ifsmall:
            low = get_shifted_int(nbits)
            builder.store(builder.zext(low, int64_t), ret)
        with iflarge:
            if is_numpy:
                high = get_shifted_int(builder.sub(nbits, c32))
            low = get_next_int32(context, builder, state_ptr)
            if not is_numpy:
                high = get_shifted_int(builder.sub(nbits, c32))
            total = builder.add(builder.zext(low, int64_t), builder.shl(builder.zext(high, int64_t), ir.Constant(int64_t, 32)))
            builder.store(total, ret)
    return builder.load(ret)