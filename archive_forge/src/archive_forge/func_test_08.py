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
def test_08(self):

    @njit
    def foo(tup1, tup2):
        acc = 0
        for a in literal_unroll(tup1):
            if a == 'a':
                acc += tup2[0]()
            elif a == 'b':
                acc += tup2[1]()
            elif a == 'c':
                acc += tup2[2]()
        return acc

    def gen(x):

        def impl():
            return x
        return njit(impl)
    tup1 = ('a', 'b', 'c', 12, 3j, ('f',))
    tup2 = (gen(1), gen(2), gen(3))
    self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))