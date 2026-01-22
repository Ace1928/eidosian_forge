import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_spacing_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    dispatch_table = {types.float32: 'numba_nextafterf', types.float64: 'numba_nextafter'}
    [ty] = sig.args
    inner_sig = typing.signature(sig.return_type, ty, ty)
    ll_ty = args[0].type
    ll_inf = ll_ty(np.inf)
    fnty = llvmlite.ir.FunctionType(ll_ty, [ll_ty, ll_ty])
    fn = cgutils.insert_pure_function(builder.module, fnty, name='llvm.copysign')
    ll_sinf = builder.call(fn, [ll_inf, args[0]])
    inner_args = args + [ll_sinf]
    nextafter = _dispatch_func_by_name_type(context, builder, inner_sig, inner_args, dispatch_table, 'nextafter')
    return builder.fsub(nextafter, args[0])