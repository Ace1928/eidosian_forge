import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@overload(np.geomspace)
def numpy_geomspace(start, stop, num=50):
    if not isinstance(start, types.Number):
        msg = 'The argument "start" must be a number'
        raise errors.TypingError(msg)
    if not isinstance(stop, types.Number):
        msg = 'The argument "stop" must be a number'
        raise errors.TypingError(msg)
    if not isinstance(num, (int, types.Integer)):
        msg = 'The argument "num" must be an integer'
        raise errors.TypingError(msg)
    if any((isinstance(arg, types.Complex) for arg in [start, stop])):
        result_dtype = from_dtype(np.result_type(as_dtype(start), as_dtype(stop), None))

        def impl(start, stop, num=50):
            if start == 0 or stop == 0:
                raise ValueError('Geometric sequence cannot include zero')
            start = result_dtype(start)
            stop = result_dtype(stop)
            both_imaginary = (start.real == 0) & (stop.real == 0)
            both_negative = (np.sign(start) == -1) & (np.sign(stop) == -1)
            out_sign = 1
            if both_imaginary:
                start = start.imag
                stop = stop.imag
                out_sign = 1j
            if both_negative:
                start = -start
                stop = -stop
                out_sign = -out_sign
            logstart = np.log10(start)
            logstop = np.log10(stop)
            result = np.logspace(logstart, logstop, num)
            if num > 0:
                result[0] = start
                if num > 1:
                    result[-1] = stop
            return out_sign * result
    else:

        def impl(start, stop, num=50):
            if start == 0 or stop == 0:
                raise ValueError('Geometric sequence cannot include zero')
            both_negative = (np.sign(start) == -1) & (np.sign(stop) == -1)
            out_sign = 1
            if both_negative:
                start = -start
                stop = -stop
                out_sign = -out_sign
            logstart = np.log10(start)
            logstop = np.log10(stop)
            result = np.logspace(logstart, logstop, num)
            if num > 0:
                result[0] = start
                if num > 1:
                    result[-1] = stop
            return out_sign * result
    return impl