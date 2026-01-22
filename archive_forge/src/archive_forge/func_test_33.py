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
def test_33(self):

    @njit
    def consumer(func, arg):
        yield func(arg)

    def get(cons):

        @njit
        def foo():

            def gen(a):
                for x in literal_unroll(a):
                    yield x
            return [next(x) for x in cons(gen, (1, 2.3, 4j))]
        return foo
    cfunc = get(consumer)
    pyfunc = get(consumer.py_func).py_func
    self.assertEqual(cfunc(), pyfunc())