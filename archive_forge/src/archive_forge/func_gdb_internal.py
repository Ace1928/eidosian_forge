import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic
@intrinsic
def gdb_internal(tyctx):
    function_sig = types.void()

    def codegen(cgctx, builder, signature, args):
        init_gdb_codegen(cgctx, builder, signature, args, const_args, do_break=do_break)
        return cgctx.get_constant(types.none, None)
    return (function_sig, codegen)