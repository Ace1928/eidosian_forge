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
def select_known_backend(backend):
    """
                Loads a specific threading layer backend based on string
                """
    lib = None
    if backend.startswith('tbb'):
        try:
            _check_tbb_version_compatible()
            from numba.np.ufunc import tbbpool as lib
        except ImportError:
            pass
    elif backend.startswith('omp'):
        try:
            from numba.np.ufunc import omppool as lib
        except ImportError:
            pass
    elif backend.startswith('workqueue'):
        from numba.np.ufunc import workqueue as lib
    else:
        msg = 'Unknown value specified for threading layer: %s'
        raise ValueError(msg % backend)
    return lib