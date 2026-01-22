import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
def test_stencilcall(self):
    from numba.stencils.stencil import stencil

    @stencil
    def kernel_1(a):
        return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])

    def test_1(n):
        a = np.ones((n, n))
        b = kernel_1(a)
        return a + b
    self._compile_and_test(test_1, (types.intp,), equivs=[self.with_equiv('a', 'b')], asserts=[self.without_assert('a', 'b')])

    def test_2(n):
        a = np.ones((n, n))
        b = np.ones((n + 1, n + 1))
        kernel_1(a, out=b)
        return a
    self._compile_and_test(test_2, (types.intp,), equivs=[self.without_equiv('a', 'b')])

    @stencil(standard_indexing=('c',))
    def kernel_2(a, b, c):
        return a[0, 1, 0] + b[0, -1, 0] + c[0]

    def test_3(n):
        a = np.arange(64).reshape(4, 8, 2)
        b = np.arange(64).reshape(n, 8, 2)
        u = np.zeros(1)
        v = kernel_2(a, b, u)
        return v
    self._compile_and_test(test_3, (types.intp,), equivs=[self.with_equiv('a', 'b', 'v'), self.without_equiv('a', 'u')], asserts=[self.with_assert('a', 'b')])