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
def test_with_kwargs2(self):

    @njit(inline='always')
    def bar(a, b=12, c=9):
        return a + b

    def impl(a, b=7, c=5):
        return bar(a + b, c=19)
    self.check(impl, 3, 4, inline_expect={'bar': True})