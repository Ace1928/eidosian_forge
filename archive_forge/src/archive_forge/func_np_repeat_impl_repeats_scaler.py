import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@register_jitable
def np_repeat_impl_repeats_scaler(a, repeats):
    if repeats < 0:
        raise ValueError('negative dimensions are not allowed')
    asa = np.asarray(a)
    aravel = asa.ravel()
    n = aravel.shape[0]
    if repeats == 0:
        return np.empty(0, dtype=asa.dtype)
    elif repeats == 1:
        return np.copy(aravel)
    else:
        to_return = np.empty(n * repeats, dtype=asa.dtype)
        for i in range(n):
            to_return[i * repeats:(i + 1) * repeats] = aravel[i]
        return to_return