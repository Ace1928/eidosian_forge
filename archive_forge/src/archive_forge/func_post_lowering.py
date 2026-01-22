import platform
import llvmlite.binding as ll
from llvmlite import ir
from numba import _dynfunc
from numba.core.callwrapper import PyCallWrapper
from numba.core.base import BaseContext
from numba.core import (utils, types, config, cgutils, callconv, codegen,
from numba.core.options import TargetOptions, include_default_options
from numba.core.runtime import rtsys
from numba.core.compiler_lock import global_compiler_lock
import numba.core.entrypoints
from numba.core.cpu_options import (ParallelOptions, # noqa F401
from numba.np import ufunc_db
def post_lowering(self, mod, library):
    if self.fastmath:
        fastmathpass.rewrite_module(mod, self.fastmath)
    if self.is32bit:
        intrinsics.fix_divmod(mod)
    library.add_linking_library(rtsys.library)