import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_ufunc_at_boolean(self):
    a = np.arange(10)
    index = a % 2 == 0
    np.equal.at(a, index, [0, 2, 4, 6, 8])
    assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])
    a = np.arange(10, dtype='u4')
    np.invert.at(a, [2, 5, 2])
    assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 4294967295, 6, 7, 8, 9])