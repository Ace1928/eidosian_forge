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
def datetime_max_impl(context, builder, sig, args):
    in1, in2 = args
    in1_not_nat = is_not_nat(builder, in1)
    in2_not_nat = is_not_nat(builder, in2)
    in1_ge_in2 = builder.icmp_signed('>=', in1, in2)
    res = builder.select(in1_ge_in2, in1, in2)
    if NAT_DOMINATES:
        in1, in2 = (in2, in1)
    res = builder.select(in1_not_nat, res, in2)
    res = builder.select(in2_not_nat, res, in1)
    return impl_ret_untracked(context, builder, sig.return_type, res)