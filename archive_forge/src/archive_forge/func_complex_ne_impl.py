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
def complex_ne_impl(context, builder, sig, args):
    [cx, cy] = args
    typ = sig.args[0]
    x = context.make_complex(builder, typ, value=cx)
    y = context.make_complex(builder, typ, value=cy)
    reals_are_ne = builder.fcmp_unordered('!=', x.real, y.real)
    imags_are_ne = builder.fcmp_unordered('!=', x.imag, y.imag)
    res = builder.or_(reals_are_ne, imags_are_ne)
    return impl_ret_untracked(context, builder, sig.return_type, res)