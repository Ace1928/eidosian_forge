from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
def test_21(self):

    @njit
    def foo(z):
        b = (23, 23.9, 6j, 8)

        def bar():
            acc = 0
            for j in literal_unroll(b):
                acc += j
            return acc
        outer_acc = 0
        for x in (1, 2, 3, 4):
            outer_acc += bar() + x
        return outer_acc
    f = 9
    k = f
    self.assertEqual(foo(k), foo.py_func(k))