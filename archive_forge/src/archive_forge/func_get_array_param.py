import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def get_array_param(ty, shapes, data):
    return (notrans if ty.layout == out_type.layout else trans, shapes[1] if ty.layout == 'C' else shapes[0], builder.bitcast(data, ll_void_p))