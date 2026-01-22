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
@lower_constant(types.Integer)
@lower_constant(types.Float)
@lower_constant(types.Boolean)
def constant_integer(context, builder, ty, pyval):
    if isinstance(pyval, np.bool_):
        pyval = bool(pyval)
    lty = context.get_value_type(ty)
    return lty(pyval)