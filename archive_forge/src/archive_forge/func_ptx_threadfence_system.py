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
@lower(stubs.threadfence_system)
def ptx_threadfence_system(context, builder, sig, args):
    assert not args
    fname = 'llvm.nvvm.membar.sys'
    lmod = builder.module
    fnty = ir.FunctionType(ir.VoidType(), ())
    sync = cgutils.get_or_insert_function(lmod, fnty, fname)
    builder.call(sync, ())
    return context.get_dummy_value()