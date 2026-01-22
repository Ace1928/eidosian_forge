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
def test_inline_w_freevar_from_another_module(self):
    from .inlining_usecases import bop_factory

    def gen(a, b):
        bar = bop_factory(a)

        def impl():
            z = _GLOBAL1 + a * b
            return (bar(), z, a)
        return impl
    impl = gen(10, 20)
    self.check(impl, inline_expect={'bar': True})