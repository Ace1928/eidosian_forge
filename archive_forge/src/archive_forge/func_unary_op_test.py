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
def unary_op_test(self, operator, nrt=True, skip_inputs=[], additional_inputs=[], int_output_type=None, float_output_type=None):
    operator_func = _make_unary_ufunc_op_usecase(operator)
    inputs = list(self.inputs)
    inputs.extend(additional_inputs)
    pyfunc = operator_func
    for input_tuple in inputs:
        input_operand, input_type = input_tuple
        if input_type in skip_inputs or not isinstance(input_type, types.Array):
            continue
        cfunc = self._compile(pyfunc, (input_type,), nrt=nrt)
        expected = pyfunc(input_operand)
        got = cfunc(input_operand)
        self._check_results(expected, got)