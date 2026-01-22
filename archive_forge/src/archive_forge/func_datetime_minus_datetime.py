import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
@lower_builtin(operator.sub, types.NPDatetime, types.NPDatetime)
def datetime_minus_datetime(context, builder, sig, args):
    va, vb = args
    ta, tb = sig.args
    unit_a = ta.unit
    unit_b = tb.unit
    ret_unit = sig.return_type.unit
    ret = alloc_timedelta_result(builder)
    with cgutils.if_likely(builder, are_not_nat(builder, [va, vb])):
        va = convert_datetime_for_arith(builder, va, unit_a, ret_unit)
        vb = convert_datetime_for_arith(builder, vb, unit_b, ret_unit)
        ret_val = builder.sub(va, vb)
        builder.store(ret_val, ret)
    res = builder.load(ret)
    return impl_ret_untracked(context, builder, sig.return_type, res)