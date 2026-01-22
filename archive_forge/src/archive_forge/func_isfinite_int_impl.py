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
@lower(math.isfinite, types.Integer)
def isfinite_int_impl(context, builder, sig, args):
    res = cgutils.true_bit
    return impl_ret_untracked(context, builder, sig.return_type, res)