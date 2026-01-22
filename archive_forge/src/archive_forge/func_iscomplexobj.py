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
@overload(np.iscomplexobj)
def iscomplexobj(x):
    dt = determine_dtype(x)
    if isinstance(x, types.Optional):
        dt = determine_dtype(x.type)
    iscmplx = np.issubdtype(dt, np.complexfloating)
    if isinstance(x, types.Optional):

        def impl(x):
            if x is None:
                return False
            return iscmplx
    else:

        def impl(x):
            return iscmplx
    return impl