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
def test_reduceat_shifting_sum(self):
    L = 6
    x = np.arange(L)
    idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
    assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])