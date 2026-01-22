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
def get_env_manager(self, builder, return_pyobject=False):
    envgv = self.declare_env_global(builder.module, self.get_env_name(self.fndesc))
    envarg = builder.load(envgv)
    pyapi = self.get_python_api(builder)
    pyapi.emit_environment_sentry(envarg, return_pyobject=return_pyobject, debug_msg=self.fndesc.env_name)
    env_body = self.get_env_body(builder, envarg)
    return pyapi.get_env_manager(self.environment, env_body, envarg)