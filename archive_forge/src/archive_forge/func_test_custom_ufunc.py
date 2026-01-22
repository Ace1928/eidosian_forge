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
def test_custom_ufunc(self):
    a = np.array([_rational_tests.rational(1, 2), _rational_tests.rational(1, 3), _rational_tests.rational(1, 4)], dtype=_rational_tests.rational)
    b = np.array([_rational_tests.rational(1, 2), _rational_tests.rational(1, 3), _rational_tests.rational(1, 4)], dtype=_rational_tests.rational)
    result = _rational_tests.test_add_rationals(a, b)
    expected = np.array([_rational_tests.rational(1), _rational_tests.rational(2, 3), _rational_tests.rational(1, 2)], dtype=_rational_tests.rational)
    assert_equal(result, expected)