import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
def compile_device(self, args, return_type=None):
    """Compile the device function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
    if args not in self.overloads:
        with self._compiling_counter:
            debug = self.targetoptions.get('debug')
            lineinfo = self.targetoptions.get('lineinfo')
            inline = self.targetoptions.get('inline')
            fastmath = self.targetoptions.get('fastmath')
            nvvm_options = {'opt': 3 if self.targetoptions.get('opt') else 0, 'fastmath': fastmath}
            cc = get_current_device().compute_capability
            cres = compile_cuda(self.py_func, return_type, args, debug=debug, lineinfo=lineinfo, inline=inline, fastmath=fastmath, nvvm_options=nvvm_options, cc=cc)
            self.overloads[args] = cres
            cres.target_context.insert_user_function(cres.entry_point, cres.fndesc, [cres.library])
    else:
        cres = self.overloads[args]
    return cres