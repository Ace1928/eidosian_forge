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
@overload(_copy_to_fortran_order)
def ol_copy_to_fortran_order(a):
    F_layout = a.layout == 'F'
    A_layout = a.layout == 'A'

    def impl(a):
        if F_layout:
            acpy = np.copy(a)
        elif A_layout:
            flag_f = a.flags.f_contiguous
            if flag_f:
                acpy = np.copy(a.T).T
            else:
                acpy = np.asfortranarray(a)
        else:
            acpy = np.asfortranarray(a)
        return acpy
    return impl