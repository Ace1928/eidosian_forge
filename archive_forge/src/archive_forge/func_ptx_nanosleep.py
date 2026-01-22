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
@lower(stubs.nanosleep, types.uint32)
def ptx_nanosleep(context, builder, sig, args):
    nanosleep = ir.InlineAsm(ir.FunctionType(ir.VoidType(), [ir.IntType(32)]), 'nanosleep.u32 $0;', 'r', side_effect=True)
    ns = args[0]
    builder.call(nanosleep, [ns])