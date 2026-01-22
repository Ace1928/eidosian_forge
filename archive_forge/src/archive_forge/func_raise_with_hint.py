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
def raise_with_hint(required):
    errmsg = 'No threading layer could be loaded.\n%s'
    hintmsg = 'HINT:\n%s'
    if len(required) == 0:
        hint = ''
    if len(required) == 1:
        hint = hintmsg % err_helpers[required[0]]
    if len(required) > 1:
        options = '\nOR\n'.join([err_helpers[x] for x in required])
        hint = hintmsg % ('One of:\n%s' % options)
    raise ValueError(errmsg % hint)