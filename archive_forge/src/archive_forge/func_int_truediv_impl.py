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
@lower_builtin(operator.truediv, types.Integer, types.Integer)
@lower_builtin(operator.itruediv, types.Integer, types.Integer)
def int_truediv_impl(context, builder, sig, args):
    [va, vb] = args
    [ta, tb] = sig.args
    a = context.cast(builder, va, ta, sig.return_type)
    b = context.cast(builder, vb, tb, sig.return_type)
    with cgutils.if_zero(builder, b):
        context.error_model.fp_zero_division(builder, ('division by zero',))
    res = builder.fdiv(a, b)
    return impl_ret_untracked(context, builder, sig.return_type, res)