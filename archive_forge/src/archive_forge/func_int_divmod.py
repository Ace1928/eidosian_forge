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
def int_divmod(context, builder, ty, x, y):
    """
    Integer divmod(x, y).  The caller must ensure that y != 0.
    """
    if ty.signed:
        return int_divmod_signed(context, builder, ty, x, y)
    else:
        return (builder.udiv(x, y), builder.urem(x, y))