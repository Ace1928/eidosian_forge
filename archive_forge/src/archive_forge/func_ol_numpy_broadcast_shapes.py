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
@overload(np.broadcast_shapes)
def ol_numpy_broadcast_shapes(*args):
    for idx, arg in enumerate(args):
        is_int = isinstance(arg, types.Integer)
        is_int_tuple = isinstance(arg, types.UniTuple) and isinstance(arg.dtype, types.Integer)
        is_empty_tuple = isinstance(arg, types.Tuple) and len(arg.types) == 0
        if not (is_int or is_int_tuple or is_empty_tuple):
            msg = f'Argument {idx} must be either an int or tuple[int]. Got {arg}'
            raise errors.TypingError(msg)
    m = 0
    for arg in args:
        if isinstance(arg, types.Integer):
            m = max(m, 1)
        elif isinstance(arg, types.BaseTuple):
            m = max(m, len(arg))
    if m == 0:
        return lambda *args: ()
    else:
        tup_init = (1,) * m

        def impl(*args):
            r = [1] * m
            tup = tup_init
            for arg in literal_unroll(args):
                if isinstance(arg, tuple) and len(arg) > 0:
                    numpy_broadcast_shapes_list(r, m, arg)
                elif isinstance(arg, int):
                    numpy_broadcast_shapes_list(r, m, (arg,))
            for idx, elem in enumerate(r):
                tup = tuple_setitem(tup, idx, elem)
            return tup
        return impl