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
def specialize(self, *args):
    """
        Create a new instance of this dispatcher specialized for the given
        *args*.
        """
    cc = get_current_device().compute_capability
    argtypes = tuple([self.typingctx.resolve_argument_type(a) for a in args])
    if self.specialized:
        raise RuntimeError('Dispatcher already specialized')
    specialization = self.specializations.get((cc, argtypes))
    if specialization:
        return specialization
    targetoptions = self.targetoptions
    specialization = CUDADispatcher(self.py_func, targetoptions=targetoptions)
    specialization.compile(argtypes)
    specialization.disable_compile()
    specialization._specialized = True
    self.specializations[cc, argtypes] = specialization
    return specialization