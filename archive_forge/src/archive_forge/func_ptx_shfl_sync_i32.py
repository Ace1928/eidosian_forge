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
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.i8, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f4, types.i4, types.i4)
@lower(stubs.shfl_sync_intrinsic, types.i4, types.i4, types.f8, types.i4, types.i4)
def ptx_shfl_sync_i32(context, builder, sig, args):
    """
    The NVVM intrinsic for shfl only supports i32, but the cuda intrinsic
    function supports both 32 and 64 bit ints and floats, so for feature parity,
    i64, f32, and f64 are implemented. Floats by way of bitcasting the float to
    an int, then shuffling, then bitcasting back. And 64-bit values by packing
    them into 2 32bit values, shuffling thoose, and then packing back together.
    """
    mask, mode, value, index, clamp = args
    value_type = sig.args[2]
    if value_type in types.real_domain:
        value = builder.bitcast(value, ir.IntType(value_type.bitwidth))
    fname = 'llvm.nvvm.shfl.sync.i32'
    lmod = builder.module
    fnty = ir.FunctionType(ir.LiteralStructType((ir.IntType(32), ir.IntType(1))), (ir.IntType(32), ir.IntType(32), ir.IntType(32), ir.IntType(32), ir.IntType(32)))
    func = cgutils.get_or_insert_function(lmod, fnty, fname)
    if value_type.bitwidth == 32:
        ret = builder.call(func, (mask, mode, value, index, clamp))
        if value_type == types.float32:
            rv = builder.extract_value(ret, 0)
            pred = builder.extract_value(ret, 1)
            fv = builder.bitcast(rv, ir.FloatType())
            ret = cgutils.make_anonymous_struct(builder, (fv, pred))
    else:
        value1 = builder.trunc(value, ir.IntType(32))
        value_lshr = builder.lshr(value, context.get_constant(types.i8, 32))
        value2 = builder.trunc(value_lshr, ir.IntType(32))
        ret1 = builder.call(func, (mask, mode, value1, index, clamp))
        ret2 = builder.call(func, (mask, mode, value2, index, clamp))
        rv1 = builder.extract_value(ret1, 0)
        rv2 = builder.extract_value(ret2, 0)
        pred = builder.extract_value(ret1, 1)
        rv1_64 = builder.zext(rv1, ir.IntType(64))
        rv2_64 = builder.zext(rv2, ir.IntType(64))
        rv_shl = builder.shl(rv2_64, context.get_constant(types.i8, 32))
        rv = builder.or_(rv_shl, rv1_64)
        if value_type == types.float64:
            rv = builder.bitcast(rv, ir.DoubleType())
        ret = cgutils.make_anonymous_struct(builder, (rv, pred))
    return ret