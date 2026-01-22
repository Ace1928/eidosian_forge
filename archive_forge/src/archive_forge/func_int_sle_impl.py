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
def int_sle_impl(context, builder, sig, args):
    res = builder.icmp_signed('<=', *args)
    return impl_ret_untracked(context, builder, sig.return_type, res)