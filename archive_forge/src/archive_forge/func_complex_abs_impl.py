import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def complex_abs_impl(context, builder, sig, args):
    """
    abs(z) := hypot(z.real, z.imag)
    """

    def complex_abs(z):
        return math.hypot(z.real, z.imag)
    res = context.compile_internal(builder, complex_abs, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)