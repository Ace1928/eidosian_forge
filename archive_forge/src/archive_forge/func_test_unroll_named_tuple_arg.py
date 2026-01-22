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
def test_unroll_named_tuple_arg(self):
    ABC = namedtuple('ABC', ['a', 'b', 'c'])

    @njit
    def foo(x):
        out = 0
        for i in literal_unroll(x):
            out += i
        return out
    abc = ABC(1, 2j, 3.4)
    self.assertEqual(foo(abc), foo.py_func(abc))