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
def test_09(self):

    @njit
    def foo(tup1, tup2):
        acc = 0
        idx = 0
        for a in literal_unroll(tup1):
            if a == 'a':
                acc += tup2[idx]
            elif a == 'b':
                acc += tup2[idx]
            elif a == 'c':
                acc += tup2[idx]
            idx += 1
        return (idx, acc)

    @njit
    def func1():
        return 1

    @njit
    def func2():
        return 2

    @njit
    def func3():
        return 3
    tup1 = ('a', 'b', 'c')
    tup2 = (1j, 1, 2)
    with self.assertRaises(errors.TypingError) as raises:
        foo(tup1, tup2)
    self.assertIn(_header_lead, str(raises.exception))