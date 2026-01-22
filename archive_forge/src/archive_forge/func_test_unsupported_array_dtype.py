import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def test_unsupported_array_dtype(self):
    cfunc = jit(nopython=True)(nop)
    a = np.ones(3)
    a = a.astype(a.dtype.newbyteorder())
    with self.assertRaises(TypingError) as raises:
        cfunc(1, a, a)
    expected = f'Unsupported array dtype: {a.dtype}'
    self.assertIn(expected, str(raises.exception))