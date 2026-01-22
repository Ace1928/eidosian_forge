import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def test_comp_unsupported_iter(self):

    def comp_unsupported_iter():
        val = zip([1, 2, 3], [4, 5, 6])
        return np.array([a for a, b in val])
    with self.assertRaises(TypingError) as raises:
        self.check(comp_unsupported_iter)
    self.assertIn(_header_lead, str(raises.exception))
    self.assertIn('Unsupported iterator found in array comprehension', str(raises.exception))