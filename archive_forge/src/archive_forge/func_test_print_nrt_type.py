import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
@unittest.skip('Issue with intermittent NRT leak, see #9355.')
def test_print_nrt_type(self):
    with self.assertNoNRTLeak():
        x = [1, 3, 5, 7]
        with self.assertRefCount(x):
            self.check_values(types.List(types.intp, reflected=True), (x,))