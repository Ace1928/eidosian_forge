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
def test_empty_list_not_removed(self):

    def f(x):
        t = []
        myList = np.array([1])
        a = np.random.choice(myList, 1)
        t.append(x + a)
        return a
    self.check(f, 5, assert_allocate_list=True)