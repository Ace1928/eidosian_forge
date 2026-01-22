import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
@lower(math.isfinite, types.Integer)
def math_isfinite_int(context, builder, sig, args):
    return context.get_constant(types.boolean, 1)