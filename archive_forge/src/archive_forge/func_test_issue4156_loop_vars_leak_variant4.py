import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase
def test_issue4156_loop_vars_leak_variant4(self):
    """Variant of test_issue4156_loop_vars_leak.

        Interleaves loops and allocations
        """

    @njit
    def udt(N):
        sum_vec = 0
        for n in range(N):
            vec = np.zeros(7)
            for n in range(N):
                z = np.zeros(7)
            sum_vec += vec[0] + z[0]
        return sum_vec
    got = udt(4)
    expect = udt.py_func(4)
    self.assertPreciseEqual(got, expect)