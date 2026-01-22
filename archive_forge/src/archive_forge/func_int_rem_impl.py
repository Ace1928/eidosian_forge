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
@lower_builtin(operator.mod, types.Integer, types.Integer)
@lower_builtin(operator.imod, types.Integer, types.Integer)
def int_rem_impl(context, builder, sig, args):
    quot, rem = _int_divmod_impl(context, builder, sig, args, 'integer modulo by zero')
    return builder.load(rem)