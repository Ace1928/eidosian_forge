import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic
@overload(gdb)
def hook_gdb(*args):
    _confirm_gdb()
    gdbimpl = gen_gdb_impl(args, True)

    def impl(*args):
        gdbimpl()
    return impl