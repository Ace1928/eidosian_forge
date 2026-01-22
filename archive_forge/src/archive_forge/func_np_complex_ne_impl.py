import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_ne_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2, return_type=types.boolean)
    ty = sig.args[0]
    in1, in2 = [context.make_complex(builder, ty, value=arg) for arg in args]
    xr = in1.real
    xi = in1.imag
    yr = in2.real
    yi = in2.imag
    xr_ne_yr = builder.fcmp_unordered('!=', xr, yr)
    xi_ne_yi = builder.fcmp_unordered('!=', xi, yi)
    return builder.or_(xr_ne_yr, xi_ne_yi)