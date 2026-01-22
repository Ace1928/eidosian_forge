import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase
def test_issue4156_loop_vars_leak(self):
    """Test issues with zero-filling of refct'ed variables inside loops.

        Before the fix, the in-loop variables are always zero-filled at their
        definition location. As a result, their state from the previous
        iteration is erased. No decref is applied. To fix this, the
        zero-filling must only happen once after the alloca at the function
        entry block. The loop variables are technically defined once per
        function (one alloca per definition per function), but semantically
        defined once per assignment. Semantically, their lifetime stop only
        when the variable is re-assigned or when the function ends.
        """

    @njit
    def udt(N):
        sum_vec = np.zeros(3)
        for n in range(N):
            if n >= 0:
                vec = np.ones(1)
            if n >= 0:
                sum_vec += vec[0]
        return sum_vec
    got = udt(4)
    expect = udt.py_func(4)
    self.assertPreciseEqual(got, expect)