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
def test_31(self):

    @njit
    def foo():
        const = 1234

        def bar(t):
            acc = 0
            a = (12, 12.7, 3j, 4)
            for x in literal_unroll(a):
                acc += x + const
            return (acc, t)
        return [x for x in map(bar, (1, 2j))]
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    self.assertIn(_header_lead, str(raises.exception))
    self.assertIn('zip', str(raises.exception))