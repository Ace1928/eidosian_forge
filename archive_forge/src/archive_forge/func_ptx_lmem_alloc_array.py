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
@lower(cuda.local.array, types.Tuple, types.Any)
@lower(cuda.local.array, types.UniTuple, types.Any)
def ptx_lmem_alloc_array(context, builder, sig, args):
    shape = [s.literal_value for s in sig.args[0]]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=shape, dtype=dtype, symbol_name='_cudapy_lmem', addrspace=nvvm.ADDRSPACE_LOCAL, can_dynsized=False)