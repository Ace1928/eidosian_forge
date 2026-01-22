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
def call_xxgemm(context, builder, x_type, x_shapes, x_data, y_type, y_shapes, y_data, out_type, out_shapes, out_data):
    """
    Call the BLAS matrix * matrix product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32), [ll_char, ll_char, ll_char, intp_t, intp_t, intp_t, ll_void_p, ll_void_p, intp_t, ll_void_p, intp_t, ll_void_p, ll_void_p, intp_t])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_xxgemm')
    m, k = x_shapes
    _k, n = y_shapes
    dtype = x_type.dtype
    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)
    trans = ir.Constant(ll_char, ord('t'))
    notrans = ir.Constant(ll_char, ord('n'))

    def get_array_param(ty, shapes, data):
        return (notrans if ty.layout == out_type.layout else trans, shapes[1] if ty.layout == 'C' else shapes[0], builder.bitcast(data, ll_void_p))
    transa, lda, data_a = get_array_param(y_type, y_shapes, y_data)
    transb, ldb, data_b = get_array_param(x_type, x_shapes, x_data)
    _, ldc, data_c = get_array_param(out_type, out_shapes, out_data)
    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    res = builder.call(fn, (kind_val, transa, transb, n, m, k, builder.bitcast(alpha, ll_void_p), data_a, lda, data_b, ldb, builder.bitcast(beta, ll_void_p), data_c, ldc))
    check_blas_return(context, builder, res)