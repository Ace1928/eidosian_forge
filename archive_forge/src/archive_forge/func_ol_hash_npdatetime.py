import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
@overload_method(types.NPTimedelta, '__hash__')
@overload_method(types.NPDatetime, '__hash__')
def ol_hash_npdatetime(x):
    if IS_32BITS:

        def impl(x):
            x = np.int64(x)
            if x < 2 ** 31 - 1:
                y = np.int32(x)
            else:
                hi = (np.int64(x) & 18446744069414584320) >> 32
                lo = np.int64(x) & 4294967295
                y = np.int32(lo + 1000003 * hi)
            if y == -1:
                y = np.int32(-2)
            return y
    else:

        def impl(x):
            if np.int64(x) == -1:
                return np.int64(-2)
            return np.int64(x)
    return impl