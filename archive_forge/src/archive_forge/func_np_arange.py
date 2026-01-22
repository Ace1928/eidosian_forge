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
@overload(np.arange)
def np_arange(start, stop=None, step=None, dtype=None):
    if isinstance(stop, types.Optional):
        stop = stop.type
    if isinstance(step, types.Optional):
        step = step.type
    if isinstance(dtype, types.Optional):
        dtype = dtype.type
    if stop is None:
        stop = types.none
    if step is None:
        step = types.none
    if dtype is None:
        dtype = types.none
    if not isinstance(start, types.Number) or not isinstance(stop, (types.NoneType, types.Number)) or (not isinstance(step, (types.NoneType, types.Number))) or (not isinstance(dtype, (types.NoneType, types.DTypeSpec))):
        return
    if isinstance(dtype, types.NoneType):
        true_dtype = _arange_dtype(start, stop, step)
    else:
        true_dtype = dtype.dtype
    use_complex = any([isinstance(x, types.Complex) for x in (start, stop, step)])
    start_value = getattr(start, 'literal_value', None)
    stop_value = getattr(stop, 'literal_value', None)
    step_value = getattr(step, 'literal_value', None)

    def impl(start, stop=None, step=None, dtype=None):
        lit_start = start_value if start_value is not None else start
        lit_stop = stop_value if stop_value is not None else stop
        lit_step = step_value if step_value is not None else step
        _step = lit_step if lit_step is not None else 1
        if lit_stop is None:
            _start, _stop = (0, lit_start)
        else:
            _start, _stop = (lit_start, lit_stop)
        if _step == 0:
            raise ValueError('Maximum allowed size exceeded')
        nitems_c = (_stop - _start) / _step
        nitems_r = int(math.ceil(nitems_c.real))
        if use_complex is True:
            nitems_i = int(math.ceil(nitems_c.imag))
            nitems = max(min(nitems_i, nitems_r), 0)
        else:
            nitems = max(nitems_r, 0)
        arr = np.empty(nitems, true_dtype)
        val = _start
        for i in range(nitems):
            arr[i] = val + i * _step
        return arr
    return impl