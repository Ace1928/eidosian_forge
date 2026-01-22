import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def lower_binary_impl(context, builder, sig, args):
    actual_libfunc = libfunc
    fast_replacement = None
    if ty == float32 and context.fastmath:
        fast_replacement = binarys_fastmath.get(libfunc.__name__)
    if fast_replacement is not None:
        actual_libfunc = getattr(libdevice, fast_replacement)
    libfunc_impl = context.get_function(actual_libfunc, typing.signature(ty, ty, ty))
    return libfunc_impl(builder, args)