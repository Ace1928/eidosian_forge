import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
@overload_method(types.Float, '__hash__')
def float_hash(val):
    if val.bitwidth == 64:

        def impl(val):
            hashed = _Py_HashDouble(val)
            return hashed
    else:

        def impl(val):
            fpextended = np.float64(_fpext(val))
            hashed = _Py_HashDouble(fpextended)
            return hashed
    return impl