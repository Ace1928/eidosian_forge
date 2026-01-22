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
@lower(stubs.atomic.sub, types.Array, types.intp, types.Any)
@lower(stubs.atomic.sub, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.sub, types.Array, types.Tuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_sub(context, builder, dtype, ptr, val):
    if dtype == types.float32:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_sub_float32(lmod), (ptr, val))
    elif dtype == types.float64:
        lmod = builder.module
        return builder.call(nvvmutils.declare_atomic_sub_float64(lmod), (ptr, val))
    else:
        return builder.atomic_rmw('sub', ptr, val, 'monotonic')