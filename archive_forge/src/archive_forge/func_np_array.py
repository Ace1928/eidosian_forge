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
def np_array(typingctx, obj, dtype):
    _check_const_str_dtype('array', dtype)
    ret = np_array_typer(typingctx, obj, dtype)
    sig = ret(obj, dtype)

    def codegen(context, builder, sig, args):
        arrty = sig.return_type
        ndim = arrty.ndim
        seqty = sig.args[0]
        seq = args[0]
        shapes = compute_sequence_shape(context, builder, ndim, seqty, seq)
        assert len(shapes) == ndim
        check_sequence_shape(context, builder, seqty, seq, shapes)
        arr = _empty_nd_impl(context, builder, arrty, shapes)
        assign_sequence_to_array(context, builder, arr.data, shapes, arr.strides, arrty, seqty, seq)
        return impl_ret_new_ref(context, builder, sig.return_type, arr._getvalue())
    return (sig, codegen)