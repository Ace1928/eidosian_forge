import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase
def test_issue4156_loop_vars_leak_variant3(self):
    """Variant of test_issue4156_loop_vars_leak.

        Adding inner loop around allocation
        """

    @njit
    def udt(N):
        sum_vec = np.zeros(3)
        for z in range(N):
            for x in range(N):
                n = x + z
                if n >= 0:
                    for y in range(N):
                        vec = np.ones(y)
                if n >= 0:
                    sum_vec += vec[0]
        return sum_vec
    got = udt(4)
    expect = udt.py_func(4)
    self.assertPreciseEqual(got, expect)