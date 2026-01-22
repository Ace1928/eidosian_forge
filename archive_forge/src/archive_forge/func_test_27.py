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
@skip_parfors_unsupported
def test_27(self):

    @njit(parallel=True)
    def foo(z):
        a = (12, 12.7, 3j, 4, z, 2 * z)
        acc = 0
        for x in literal_unroll(a):
            for k in prange(10):
                acc += 1
        return acc
    f = 9
    k = f
    self.assertEqual(foo(k), foo.py_func(k))