import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(np.trim_zeros)
def np_trim_zeros(filt, trim='fb'):
    if not isinstance(filt, types.Array):
        raise NumbaTypeError('The first argument must be an array')
    if filt.ndim > 1:
        raise NumbaTypeError('array must be 1D')
    if not isinstance(trim, (str, types.UnicodeType)):
        raise NumbaTypeError('The second argument must be a string')

    def impl(filt, trim='fb'):
        a_ = np.asarray(filt)
        first = 0
        trim = trim.lower()
        if 'f' in trim:
            for i in a_:
                if i != 0:
                    break
                else:
                    first = first + 1
        last = len(filt)
        if 'b' in trim:
            for i in a_[::-1]:
                if i != 0:
                    break
                else:
                    last = last - 1
        return a_[first:last]
    return impl