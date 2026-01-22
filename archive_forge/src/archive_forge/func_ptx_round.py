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
@lower(round, types.f4)
@lower(round, types.f8)
def ptx_round(context, builder, sig, args):
    fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(64), (ir.DoubleType(),)), '__nv_llrint')
    return builder.call(fn, [context.cast(builder, args[0], sig.args[0], types.double)])