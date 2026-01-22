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
@lower(stubs.activemask)
def ptx_activemask(context, builder, sig, args):
    activemask = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []), 'activemask.b32 $0;', '=r', side_effect=True)
    return builder.call(activemask, [])