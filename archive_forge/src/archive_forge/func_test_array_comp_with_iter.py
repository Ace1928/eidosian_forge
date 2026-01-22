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
def test_array_comp_with_iter(self):

    def array_comp(a):
        l = np.array([x * x for x in a])
        return l
    l = [1, 2, 3, 4, 5]
    self.check(array_comp, l)
    self.check(array_comp, np.array(l))
    self.check(array_comp, tuple(l))
    self.check(array_comp, typed.List(l))