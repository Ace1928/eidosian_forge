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
def get_generator_state(self, builder, genptr, return_type):
    """
        From the given *genptr* (a pointer to a _dynfunc.Generator object),
        get a pointer to its state area.
        """
    return cgutils.pointer_add(builder, genptr, _dynfunc._impl_info['offsetof_generator_state'], return_type=return_type)