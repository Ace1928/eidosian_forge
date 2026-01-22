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
def test_issue3976(self):

    def overload_this(a):
        return 'dummy'

    @njit
    def foo(a):
        if a:
            s = 5
            s = overload_this(s)
        else:
            s = 'b'
        return s

    @overload(overload_this)
    def ol(a):
        return overload_this
    self.check_func(foo, True)