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
def test_unroll_tuple_of_dict(self):

    @njit
    def foo():
        x = {}
        x['a'] = 1
        x['b'] = 2
        y = {}
        y[3] = 'c'
        y[4] = 'd'
        for it in literal_unroll((x, y)):
            for k, v in it.items():
                print(k, v)
    with captured_stdout() as stdout:
        foo()
    lines = stdout.getvalue().splitlines()
    self.assertEqual(lines, ['a 1', 'b 2', '3 c', '4 d'])