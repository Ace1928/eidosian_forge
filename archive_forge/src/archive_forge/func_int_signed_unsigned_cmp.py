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
def int_signed_unsigned_cmp(op):

    def impl(context, builder, sig, args):
        left, right = args
        cmp_zero = builder.icmp_signed('<', left, Constant(left.type, 0))
        lt_zero = builder.icmp_signed(op, left, Constant(left.type, 0))
        ge_zero = builder.icmp_unsigned(op, left, right)
        res = builder.select(cmp_zero, lt_zero, ge_zero)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    return impl