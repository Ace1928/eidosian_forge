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
def test_14(self):

    @njit
    def foo():
        x = [1000, 2000, 3000, 4000]
        acc = 0
        for a in literal_unroll(x):
            acc += a
        x.append(10)
        return acc
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    self.assertIn("Unknown attribute 'append' of type Tuple", str(raises.exception))