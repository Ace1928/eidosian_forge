import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def binary_ufunc_mixed_types_test(self, ufunc):
    ufunc_name = ufunc.__name__
    ufunc = _make_ufunc_usecase(ufunc)
    inputs1 = [(1, types.uint64), (-1, types.int64), (0.5, types.float64), (np.array([0, 1], dtype='u8'), types.Array(types.uint64, 1, 'C')), (np.array([-1, 1], dtype='i8'), types.Array(types.int64, 1, 'C')), (np.array([-0.5, 0.5], dtype='f8'), types.Array(types.float64, 1, 'C'))]
    inputs2 = inputs1
    output_types = [types.Array(types.int64, 1, 'C'), types.Array(types.float64, 1, 'C')]
    pyfunc = ufunc
    for vals in itertools.product(inputs1, inputs2, output_types):
        input1, input2, output_type = vals
        input1_operand = input1[0]
        input1_type = input1[1]
        input2_operand = input2[0]
        input2_type = input2[1]
        if ufunc_name == 'divide' and (input2_type == types.Array(types.uint32, 1, 'C') or input2_type == types.Array(types.uint64, 1, 'C')):
            continue
        if ufunc_name == 'subtract' and input1_type == types.Array(types.uint32, 1, 'C') and (input2_type == types.uint32) and types.Array(types.int64, 1, 'C'):
            continue
        if ufunc_name == 'subtract' and input1_type == types.Array(types.uint32, 1, 'C') and (input2_type == types.uint64) and types.Array(types.int64, 1, 'C'):
            continue
        if (isinstance(input1_type, types.Array) or isinstance(input2_type, types.Array)) and (not isinstance(output_type, types.Array)):
            continue
        args = (input1_type, input2_type, output_type)
        cfunc = self._compile(pyfunc, args)
        if isinstance(input1_operand, np.ndarray):
            result = np.zeros(input1_operand.size, dtype=output_type.dtype.name)
            expected = np.zeros(input1_operand.size, dtype=output_type.dtype.name)
        elif isinstance(input2_operand, np.ndarray):
            result = np.zeros(input2_operand.size, dtype=output_type.dtype.name)
            expected = np.zeros(input2_operand.size, dtype=output_type.dtype.name)
        else:
            result = np.zeros(1, dtype=output_type.dtype.name)
            expected = np.zeros(1, dtype=output_type.dtype.name)
        cfunc(input1_operand, input2_operand, result)
        pyfunc(input1_operand, input2_operand, expected)
        scalar_type = getattr(output_type, 'dtype', output_type)
        prec = 'single' if scalar_type in (types.float32, types.complex64) else 'double'
        self.assertPreciseEqual(expected, result, prec=prec)