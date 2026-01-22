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
def test_issue4693(self):

    @njit(inline='always')
    def inlining(array):
        if array.ndim != 1:
            raise ValueError('Invalid number of dimensions')
        return array

    @njit
    def fn(array):
        return inlining(array)
    fn(np.zeros(10))