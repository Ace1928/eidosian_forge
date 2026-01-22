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
def test_multiple_overloads_with_different_inline_characteristics(self):

    def bar(x):
        if isinstance(typeof(x), types.Float):
            return x + 1234
        else:
            return x + 1

    @overload(bar, inline='always')
    def bar_int_ol(x):
        if isinstance(x, types.Integer):

            def impl(x):
                return x + 1
            return impl

    @overload(bar, inline='never')
    def bar_float_ol(x):
        if isinstance(x, types.Float):

            def impl(x):
                return x + 1234
            return impl

    def always_inline_cost_model(*args):
        return True

    @overload(bar, inline=always_inline_cost_model)
    def bar_complex_ol(x):
        if isinstance(x, types.Complex):

            def impl(x):
                return x + 1
            return impl

    def impl():
        a = bar(1)
        b = bar(2.3)
        c = bar(3j)
        return a + b + c
    fir = self.check(impl, inline_expect={'bar': False}, block_count=1)
    block = next(iter(fir.blocks.items()))[1]
    calls = [x for x in block.find_exprs(op='call')]
    self.assertTrue(len(calls) == 1)
    consts = [x.value for x in block.find_insts(ir.Assign) if isinstance(getattr(x, 'value', None), ir.Const)]
    for val in consts:
        self.assertNotEqual(val.value, 1234)