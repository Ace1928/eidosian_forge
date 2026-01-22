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
@overload(np.array_equal)
def np_array_equal(a1, a2):
    if not (type_can_asarray(a1) and type_can_asarray(a2)):
        raise TypingError('Both arguments to "array_equals" must be array-like')
    accepted = (types.Boolean, types.Number)
    if isinstance(a1, accepted) and isinstance(a2, accepted):

        def impl(a1, a2):
            return a1 == a2
    else:

        def impl(a1, a2):
            a = np.asarray(a1)
            b = np.asarray(a2)
            if a.shape == b.shape:
                return np.all(a == b)
            return False
    return impl