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
def test_matrix_multiply_umath_empty(self):
    res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
    assert_array_equal(res, np.zeros((0, 0)))
    res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
    assert_array_equal(res, np.zeros((10, 10)))