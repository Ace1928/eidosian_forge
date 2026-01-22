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
@lower(math.degrees, types.Float)
def degrees_float_impl(context, builder, sig, args):
    [x] = args
    coef = context.get_constant(sig.return_type, 180 / math.pi)
    res = builder.fmul(x, coef)
    return impl_ret_untracked(context, builder, sig.return_type, res)