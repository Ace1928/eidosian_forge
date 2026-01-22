import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
@lower(math.atan2, types.uint64, types.uint64)
def atan2_u64_impl(context, builder, sig, args):
    [y, x] = args
    y = builder.uitofp(y, llvmlite.ir.DoubleType())
    x = builder.uitofp(x, llvmlite.ir.DoubleType())
    fsig = signature(types.float64, types.float64, types.float64)
    return atan2_float_impl(context, builder, fsig, (y, x))