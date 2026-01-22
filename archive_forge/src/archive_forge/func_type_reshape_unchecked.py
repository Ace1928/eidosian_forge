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
@extending.type_callable(reshape_unchecked)
def type_reshape_unchecked(context):

    def check_shape(shape):
        return isinstance(shape, types.BaseTuple) and all((isinstance(v, types.Integer) for v in shape))

    def typer(a, shape, strides):
        if not isinstance(a, types.Array):
            return
        if not check_shape(shape) or not check_shape(strides):
            return
        if len(shape) != len(strides):
            return
        return a.copy(ndim=len(shape), layout='A')
    return typer