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
def ptx_fp16_comparison(context, builder, sig, args):
    fnty = ir.FunctionType(ir.IntType(16), [ir.IntType(16), ir.IntType(16)])
    asm = ir.InlineAsm(fnty, _fp16_cmp.format(op=op), '=h,h,h')
    result = builder.call(asm, args)
    zero = context.get_constant(types.int16, 0)
    int_result = builder.bitcast(result, ir.IntType(16))
    return builder.icmp_unsigned('!=', int_result, zero)