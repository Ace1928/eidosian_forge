import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_nextafter_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    dispatch_table = {types.float32: 'numba_nextafterf', types.float64: 'numba_nextafter'}
    return _dispatch_func_by_name_type(context, builder, sig, args, dispatch_table, 'nextafter')