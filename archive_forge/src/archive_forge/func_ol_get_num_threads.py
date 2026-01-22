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
@overload(get_num_threads)
def ol_get_num_threads():
    _launch_threads()

    def impl():
        num_threads = _get_num_threads()
        if num_threads <= 0:
            print('Broken thread_id: ', get_thread_id())
            print('num_threads: ', num_threads)
            raise RuntimeError('Invalid number of threads. This likely indicates a bug in Numba.')
        return num_threads
    return impl