import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
@lower(operator.truediv, types.float32, types.float32)
def maybe_fast_truediv(context, builder, sig, args):
    if context.fastmath:
        sig = typing.signature(float32, float32, float32)
        impl = context.get_function(libdevice.fast_fdividef, sig)
        return impl(builder, args)
    else:
        with cgutils.if_zero(builder, args[1]):
            context.error_model.fp_zero_division(builder, ('division by zero',))
        res = builder.fdiv(*args)
        return res