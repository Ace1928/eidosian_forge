from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def test_debug_and_lineinfo_warning(self):
    with warnings.catch_warnings(record=True) as w:
        ignore_internal_warnings()

        @cuda.jit(debug=True, lineinfo=True, opt=False)
        def f():
            pass
    self.assertEqual(len(w), 1)
    self.assertEqual(w[0].category, NumbaInvalidConfigWarning)
    self.assertIn('debug and lineinfo are mutually exclusive', str(w[0].message))