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
def test_unary_invert_array_op(self):
    self.unary_op_test('~', skip_inputs=[types.float32, types.float64, types.Array(types.float32, 1, 'C'), types.Array(types.float64, 1, 'C')], additional_inputs=self.bitwise_additional_inputs())