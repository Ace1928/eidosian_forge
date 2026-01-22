from numba import jit, njit
from numba.core import types, ir, config, compiler
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.tests.support import IRPreservingTestPipeline
import numpy as np
import unittest
def test_input_ir_copy_remove_transform(self):
    """make sure Interpreter._remove_unused_temporaries() does not generate
        invalid code for rare chained assignment cases
        """

    def impl1(a):
        b = c = a + 1
        return (b, c)

    def impl2(A, i, a):
        b = A[i] = a + 1
        return (b, A[i] + 2)

    def impl3(A, a):
        b = A.a = a + 1
        return (b, A.a + 2)

    class C:
        pass
    self.assertEqual(impl1(5), njit(impl1)(5))
    self.assertEqual(impl2(np.ones(3), 0, 5), njit(impl2)(np.ones(3), 0, 5))
    self.assertEqual(impl3(C(), 5), jit(forceobj=True)(impl3)(C(), 5))