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
def scalar_view(scalar, viewty):
    """ Typing for the np scalar 'view' method. """
    if isinstance(scalar, (types.Float, types.Integer)) and isinstance(viewty, types.abstract.DTypeSpec):
        if scalar.bitwidth != viewty.dtype.bitwidth:
            raise errors.TypingError('Changing the dtype of a 0d array is only supported if the itemsize is unchanged')

        def impl(scalar, viewty):
            return viewer(scalar, viewty)
        return impl