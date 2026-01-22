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
def test_issue5223(self):

    @njit
    def bar(x):
        if len(x) == 5:
            return x
        x = x.copy()
        for i in range(len(x)):
            x[i] += 1
        return x
    a = np.ones(5)
    a.flags.writeable = False
    np.testing.assert_allclose(bar(a), bar.py_func(a))