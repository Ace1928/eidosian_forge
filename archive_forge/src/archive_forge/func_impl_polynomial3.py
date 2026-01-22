from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
@lower_builtin(Polynomial, types.Array, types.Array, types.Array)
def impl_polynomial3(context, builder, sig, args):

    def to_double(coef):
        return np.asarray(coef, dtype=np.double)
    typ = sig.return_type
    polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    coef_sig = sig.args[0].copy(dtype=types.double)(sig.args[0])
    domain_sig = sig.args[1].copy(dtype=types.double)(sig.args[1])
    window_sig = sig.args[2].copy(dtype=types.double)(sig.args[2])
    coef_cast = context.compile_internal(builder, to_double, coef_sig, (args[0],))
    domain_cast = context.compile_internal(builder, to_double, domain_sig, (args[1],))
    window_cast = context.compile_internal(builder, to_double, window_sig, (args[2],))
    domain_helper = context.make_helper(builder, domain_sig.return_type, value=domain_cast)
    window_helper = context.make_helper(builder, window_sig.return_type, value=window_cast)
    i64 = ir.IntType(64)
    two = i64(2)
    s1 = builder.extract_value(domain_helper.shape, 0)
    s2 = builder.extract_value(window_helper.shape, 0)
    pred1 = builder.icmp_signed('!=', s1, two)
    pred2 = builder.icmp_signed('!=', s2, two)
    with cgutils.if_unlikely(builder, pred1):
        context.call_conv.return_user_exc(builder, ValueError, ('Domain has wrong number of elements.',))
    with cgutils.if_unlikely(builder, pred2):
        context.call_conv.return_user_exc(builder, ValueError, ('Window has wrong number of elements.',))
    polynomial.coef = coef_cast
    polynomial.domain = domain_helper._getvalue()
    polynomial.window = window_helper._getvalue()
    return polynomial._getvalue()