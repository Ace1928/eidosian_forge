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
def test_inline_always_kw_no_default(self):

    def foo(a, b):
        return a + b

    @overload(foo, inline='always')
    def overload_foo(a, b):
        return lambda a, b: a + b

    def impl():
        return foo(3, b=4)
    self.check(impl, inline_expect={'foo': True})