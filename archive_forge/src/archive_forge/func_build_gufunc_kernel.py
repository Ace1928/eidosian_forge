import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
def build_gufunc_kernel(library, ctx, info, sig, inner_ndim):
    """Wrap the original CPU ufunc/gufunc with a parallel dispatcher.
    This function will wrap gufuncs and ufuncs something like.

    Args
    ----
    ctx
        numba's codegen context

    info: (library, env, name)
        inner function info

    sig
        type signature of the gufunc

    inner_ndim
        inner dimension of the gufunc (this is len(sig.args) in the case of a
        ufunc)

    Returns
    -------
    wrapper_info : (library, env, name)
        The info for the gufunc wrapper.

    Details
    -------

    The kernel signature looks like this:

    void kernel(char **args, npy_intp *dimensions, npy_intp* steps, void* data)

    args - the input arrays + output arrays
    dimensions - the dimensions of the arrays
    steps - the step size for the array (this is like sizeof(type))
    data - any additional data

    The parallel backend then stages multiple calls to this kernel concurrently
    across a number of threads. Practically, for each item of work, the backend
    duplicates `dimensions` and adjusts the first entry to reflect the size of
    the item of work, it also forms up an array of pointers into the args for
    offsets to read/write from/to with respect to its position in the items of
    work. This allows the same kernel to be used for each item of work, with
    simply adjusted reads/writes/domain sizes and is safe by virtue of the
    domain partitioning.

    NOTE: The execution backend is passed the requested thread count, but it can
    choose to ignore it (TBB)!
    """
    assert isinstance(info, tuple)
    byte_t = ir.IntType(8)
    byte_ptr_t = ir.PointerType(byte_t)
    byte_ptr_ptr_t = ir.PointerType(byte_ptr_t)
    intp_t = ctx.get_value_type(types.intp)
    intp_ptr_t = ir.PointerType(intp_t)
    fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(byte_ptr_t), ir.PointerType(intp_t), ir.PointerType(intp_t), byte_ptr_t])
    wrapperlib = ctx.codegen().create_library('parallelgufuncwrapper')
    mod = wrapperlib.create_ir_module('parallel.gufunc.wrapper')
    kernel_name = '.kernel.{}_{}'.format(id(info.env), info.name)
    lfunc = ir.Function(mod, fnty, name=kernel_name)
    bb_entry = lfunc.append_basic_block('')
    builder = ir.IRBuilder(bb_entry)
    args, dimensions, steps, data = lfunc.args
    pyapi = ctx.get_python_api(builder)
    gil_state = pyapi.gil_ensure()
    thread_state = pyapi.save_thread()

    def as_void_ptr(arg):
        return builder.bitcast(arg, byte_ptr_t)
    array_count = len(sig.args)
    if not isinstance(sig.return_type, types.NoneType):
        array_count += 1
    parallel_for_ty = ir.FunctionType(ir.VoidType(), [byte_ptr_t] * 5 + [intp_t] * 3)
    parallel_for = cgutils.get_or_insert_function(mod, parallel_for_ty, 'numba_parallel_for')
    innerfunc_fnty = ir.FunctionType(ir.VoidType(), [byte_ptr_ptr_t, intp_ptr_t, intp_ptr_t, byte_ptr_t])
    tmp_voidptr = cgutils.get_or_insert_function(mod, innerfunc_fnty, info.name)
    wrapperlib.add_linking_library(info.library)
    get_num_threads = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(types.intp.bitwidth), []), 'get_num_threads')
    num_threads = builder.call(get_num_threads, [])
    fnptr = builder.bitcast(tmp_voidptr, byte_ptr_t)
    innerargs = [as_void_ptr(x) for x in [args, dimensions, steps, data]]
    builder.call(parallel_for, [fnptr] + innerargs + [intp_t(x) for x in (inner_ndim, array_count)] + [num_threads])
    pyapi.restore_thread(thread_state)
    pyapi.gil_release(gil_state)
    builder.ret_void()
    wrapperlib.add_ir_module(mod)
    wrapperlib.add_linking_library(library)
    return _wrapper_info(library=wrapperlib, name=lfunc.name, env=info.env)