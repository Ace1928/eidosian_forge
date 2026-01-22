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
def inspect_sass_cfg(self, signature=None):
    """
        Return this kernel's CFG for the device in the current context.

        :param signature: A tuple of argument types.
        :return: The CFG for the given signature, or a dict of CFGs
                 for all previously-encountered signatures.

        The CFG for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
    if self.targetoptions.get('device'):
        raise RuntimeError('Cannot get the CFG of a device function')
    if signature is not None:
        return self.overloads[signature].inspect_sass_cfg()
    else:
        return {sig: defn.inspect_sass_cfg() for sig, defn in self.overloads.items()}