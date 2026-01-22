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
@intrinsic
def np_column_stack(typingctx, tup):
    dtype, ndim = _sequence_of_arrays(typingctx, 'np.column_stack', tup, dim_chooser=_column_stack_dims)
    layout = _choose_concatenation_layout(tup)
    ret = types.Array(dtype, ndim, layout)
    sig = ret(tup)

    def codegen(context, builder, sig, args):
        orig_arrtys = list(sig.args[0])
        orig_arrs = cgutils.unpack_tuple(builder, args[0])
        arrtys = []
        arrs = []
        axis = context.get_constant(types.intp, 1)
        for arrty, arr in zip(orig_arrtys, orig_arrs):
            if arrty.ndim == 2:
                arrtys.append(arrty)
                arrs.append(arr)
            else:
                assert arrty.ndim == 1
                newty = arrty.copy(ndim=2)
                expand_sig = typing.signature(newty, arrty)
                newarr = expand_dims(context, builder, expand_sig, (arr,), axis)
                arrtys.append(newty)
                arrs.append(newarr)
        return _np_concatenate(context, builder, arrtys, arrs, sig.return_type, axis)
    return (sig, codegen)