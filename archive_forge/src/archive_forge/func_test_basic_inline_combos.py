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
def test_basic_inline_combos(self):

    def impl():
        x = foo()
        y = bar()
        z = baz()
        return (x, y, z)
    opts = ('always', 'never')
    for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

        def foo():
            pass

        def bar():
            pass

        def baz():
            pass

        @overload(foo, inline=inline_foo)
        def foo_overload():

            def impl():
                return
            return impl

        @overload(bar, inline=inline_bar)
        def bar_overload():

            def impl():
                return
            return impl

        @overload(baz, inline=inline_baz)
        def baz_overload():

            def impl():
                return
            return impl
        inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
        self.check(impl, inline_expect=inline_expect)