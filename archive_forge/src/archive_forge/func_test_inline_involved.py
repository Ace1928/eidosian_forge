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
def test_inline_involved(self):
    fortran = njit(inline='always')(_gen_involved())

    @njit(inline='always')
    def boz(j):
        acc = 0

        def biz(t):
            return t + acc
        for x in range(j):
            acc += biz(8 + acc) + fortran(2.0, acc, 1, 12j, biz(acc))
        return acc

    @njit(inline='always')
    def foo(a):
        acc = 0
        for p in range(12):
            tmp = fortran(1, 1, 1, 1, 1)

            def baz(x):
                return 12 + a + x + tmp
            acc += baz(p) + 8 + boz(p) + tmp
        return acc + baz(2)

    def impl():
        z = 9

        def bar(x):
            return foo(z) + 7 + x
        return bar(z + 2)
    if utils.PYVERSION in ((3, 12),):
        bc = 39
    elif utils.PYVERSION in ((3, 10), (3, 11)):
        bc = 35
    elif utils.PYVERSION in ((3, 9),):
        bc = 33
    else:
        raise NotImplementedError(utils.PYVERSION)
    self.check(impl, inline_expect={'foo': True, 'boz': True, 'fortran': True}, block_count=bc)