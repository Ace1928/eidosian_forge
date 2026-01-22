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
def test_17(self):

    def dt(value):
        if value == 1000:
            return 'a'
        elif value == 2000:
            return 'b'
        elif value == 3000:
            return 'c'
        elif value == 4000:
            return 'd'
        elif value == 'f':
            return 'EFF'

    @overload(dt, inline='always')
    def ol_dt(li):
        if isinstance(li, types.IntegerLiteral):
            value = li.literal_value
            if value == 1000:

                def impl(li):
                    return 'a'
            elif value == 2000:

                def impl(li):
                    return 'b'
            elif value == 3000:

                def impl(li):
                    return 'c'
            elif value == 4000:

                def impl(li):
                    return 'd'
            return impl
        elif isinstance(li, types.StringLiteral):
            value = li.literal_value
            if value == 'f':

                def impl(li):
                    return 'EFF'
                return impl

    @njit
    def foo():
        x = (1000, 2000, 3000, 'f')
        acc = ''
        for a in literal_unroll(x[1:]):
            acc += dt(a)
        return acc
    self.assertEqual(foo(), foo.py_func())