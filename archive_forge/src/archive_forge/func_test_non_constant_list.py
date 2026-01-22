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
def test_non_constant_list(self):

    @njit
    def foo(y):
        x = [10, y]
        r = 0
        for a in literal_unroll(x):
            r += a
        return r
    with self.assertRaises(errors.UnsupportedError) as raises:
        foo(10)
    self.assertIn('Found non-constant value at position 1 in a list argument to literal_unroll', str(raises.exception))