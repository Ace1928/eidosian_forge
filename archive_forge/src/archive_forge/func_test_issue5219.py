import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
def test_issue5219(self):

    def overload_this(a, b=None):
        if isinstance(b, tuple):
            b = b[0]
        return b

    @overload(overload_this)
    def ol(a, b=None):
        b_is_tuple = isinstance(b, (types.Tuple, types.UniTuple))

        def impl(a, b=None):
            if b_is_tuple is True:
                b = b[0]
            return b
        return impl

    @njit
    def test_tuple(a, b):
        overload_this(a, b)
    self.check_func(test_tuple, 1, (2,))