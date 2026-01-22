from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
@lower(stubs.atomic.cas, types.Array, types.intp, types.Any, types.Any)
@lower(stubs.atomic.cas, types.Array, types.Tuple, types.Any, types.Any)
@lower(stubs.atomic.cas, types.Array, types.UniTuple, types.Any, types.Any)
def ptx_atomic_cas(context, builder, sig, args):
    aryty, indty, oldty, valty = sig.args
    ary, inds, old, val = args
    indty, indices = _normalize_indices(context, builder, indty, inds, aryty, valty)
    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices, wraparound=True)
    if aryty.dtype in cuda.cudadecl.integer_numba_types:
        lmod = builder.module
        bitwidth = aryty.dtype.bitwidth
        return nvvmutils.atomic_cmpxchg(builder, lmod, bitwidth, ptr, old, val)
    else:
        raise TypeError('Unimplemented atomic cas with %s array' % aryty.dtype)