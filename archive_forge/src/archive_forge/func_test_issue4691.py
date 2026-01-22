import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def test_issue4691(self):

    def output_factory(array, dtype):
        pass

    @overload(output_factory, inline='always')
    def ol_output_factory(array, dtype):
        if isinstance(array, types.npytypes.Array):

            def impl(array, dtype):
                shape = array.shape[3:]
                return np.zeros(shape, dtype=dtype)
            return impl

    @njit(nogil=True)
    def fn(array):
        out = output_factory(array, array.dtype)
        return out

    @njit(nogil=True)
    def fn2(array):
        return np.zeros(array.shape[3:], dtype=array.dtype)
    fn(np.ones((10, 20, 30, 40, 50)))
    fn2(np.ones((10, 20, 30, 40, 50)))