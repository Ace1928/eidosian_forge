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
def test_unroll_tuple_nested(self):

    @njit
    def foo():
        x = ((10, 1.2), (1j, 3.0))
        out = 0
        for i in literal_unroll(x):
            for j in i:
                out += j
        return out
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    self.assertIn('getiter', str(raises.exception))
    re = '.*Tuple\\(int[0-9][0-9], float64\\).*'
    self.assertRegex(str(raises.exception), re)