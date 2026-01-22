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
def set_parallel_chunksize(n):
    _launch_threads()
    if not isinstance(n, (int, np.integer)):
        raise TypeError('The parallel chunksize must be an integer')
    global _set_parallel_chunksize
    if n < 0:
        raise ValueError('chunksize must be greater than or equal to zero')
    return _set_parallel_chunksize(n)