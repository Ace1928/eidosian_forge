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
@lower_cast(types.Integer, types.Float)
def integer_to_float(context, builder, fromty, toty, val):
    lty = context.get_value_type(toty)
    if fromty.signed:
        return builder.sitofp(val, lty)
    else:
        return builder.uitofp(val, lty)