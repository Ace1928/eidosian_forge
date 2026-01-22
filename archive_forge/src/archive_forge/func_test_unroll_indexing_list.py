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
def test_unroll_indexing_list(self):

    @njit
    def foo(cont):
        i = 0
        acc = 0
        normal_list = [a for a in cont]
        heter_tuple = ('a', 25, 0.23, None)
        for item in literal_unroll(heter_tuple):
            acc += normal_list[i]
            i += 1
            print(item)
        return (i, acc)
    data = [j for j in range(4)]
    with captured_stdout():
        self.assertEqual(foo(data), foo.py_func(data))
    with captured_stdout() as stdout:
        foo(data)
    lines = stdout.getvalue().splitlines()
    self.assertEqual(lines, ['a', '25', '0.23', 'None'])