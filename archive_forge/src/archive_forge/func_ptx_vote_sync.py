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
@lower(stubs.vote_sync_intrinsic, types.i4, types.i4, types.boolean)
def ptx_vote_sync(context, builder, sig, args):
    fname = 'llvm.nvvm.vote.sync'
    lmod = builder.module
    fnty = ir.FunctionType(ir.LiteralStructType((ir.IntType(32), ir.IntType(1))), (ir.IntType(32), ir.IntType(32), ir.IntType(1)))
    func = cgutils.get_or_insert_function(lmod, fnty, fname)
    return builder.call(func, args)