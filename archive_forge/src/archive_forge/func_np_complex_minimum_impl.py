import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_minimum_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    ty = sig.args[0]
    bc_sig = typing.signature(types.boolean, ty)
    bcc_sig = typing.signature(types.boolean, *[ty] * 2)
    arg1, arg2 = args
    arg1_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg1])
    arg2_nan = np_complex_isnan_impl(context, builder, bc_sig, [arg2])
    any_nan = builder.or_(arg1_nan, arg2_nan)
    nan_result = builder.select(arg1_nan, arg1, arg2)
    arg1_le_arg2 = np_complex_le_impl(context, builder, bcc_sig, args)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)