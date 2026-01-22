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
def test_inline_never_operators_getitem(self):

    def impl(x, idx):
        return x[idx]

    def impl_static_getitem(x):
        return x[1]
    Dummy, DummyType = self.make_dummy_type()
    dummy_getitem_impl = lambda obj, idx: None
    setattr(Dummy, '__getitem__', dummy_getitem_impl)

    @overload(operator.getitem, inline='never')
    def overload_dummy_getitem(obj, idx):
        if isinstance(obj, DummyType):
            return dummy_getitem_impl
    self.check(impl, Dummy(), 1, inline_expect={'getitem': False})
    self.check(impl_static_getitem, Dummy(), inline_expect={'getitem': False})