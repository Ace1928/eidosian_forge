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
def test_9(self):

    @njit
    def foo(idx, z):
        a = [12, 12.7, 3j, 4]
        acc = 0
        for i in literal_unroll(a):
            acc += i
            if acc.real < 26:
                acc -= 1
            else:
                for x in literal_unroll(a):
                    acc += x
                break
        if a[0] < 23:
            acc += 2
        return acc
    f = 9
    k = f
    self.assertEqual(foo(2, k), foo.py_func(2, k))