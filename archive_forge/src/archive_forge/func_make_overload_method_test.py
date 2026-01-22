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
def make_overload_method_test(self, costmodel, should_inline):

    def costmodel(*args):
        return should_inline
    Dummy, DummyType = self.make_dummy_type()

    @overload_method(DummyType, 'inline_method', inline=costmodel)
    def _get_inlined_method(obj, val):

        def get(obj, val):
            return ('THIS IS INLINED!!!', val)
        return get

    def foo(obj):
        return obj.inline_method(123)
    self.check_method(test_impl=foo, args=[Dummy()], expected=('THIS IS INLINED!!!', 123), block_count=1, expects_inlined=should_inline)