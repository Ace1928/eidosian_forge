import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
@lower(math.sin, types.float16)
def fp16_sin_impl(context, builder, sig, args):

    def fp16_sin(x):
        return cuda.fp16.hsin(x)
    return context.compile_internal(builder, fp16_sin, sig, args)