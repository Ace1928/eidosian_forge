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
def get_env_body(self, builder, envptr):
    """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
    body_ptr = cgutils.pointer_add(builder, envptr, _dynfunc._impl_info['offsetof_env_body'])
    return EnvBody(self, builder, ref=body_ptr, cast_ref=True)