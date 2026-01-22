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
def test_inline_operators_binop(self):

    def impl_inline(x):
        return x == 1

    def impl_noinline(x):
        return x != 1
    Dummy, DummyType = self.make_dummy_type()
    dummy_binop_impl = lambda a, b: True
    setattr(Dummy, '__eq__', dummy_binop_impl)
    setattr(Dummy, '__ne__', dummy_binop_impl)

    @overload(operator.eq, inline='always')
    def overload_dummy_eq(a, b):
        if isinstance(a, DummyType):
            return dummy_binop_impl

    @overload(operator.ne, inline='never')
    def overload_dummy_ne(a, b):
        if isinstance(a, DummyType):
            return dummy_binop_impl
    self.check(impl_inline, Dummy(), inline_expect={'eq': True})
    self.check(impl_noinline, Dummy(), inline_expect={'ne': False})