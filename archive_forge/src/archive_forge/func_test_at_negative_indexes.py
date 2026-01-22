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
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'] + np.typecodes['Float'])
@pytest.mark.parametrize('ufunc', [np.add, np.subtract, np.divide, np.minimum, np.maximum])
def test_at_negative_indexes(self, dtype, ufunc):
    a = np.arange(0, 10).astype(dtype)
    indxs = np.array([-1, 1, -1, 2]).astype(np.intp)
    vals = np.array([1, 5, 2, 10], dtype=a.dtype)
    expected = a.copy()
    for i, v in zip(indxs, vals):
        expected[i] = ufunc(expected[i], v)
    ufunc.at(a, indxs, vals)
    assert_array_equal(a, expected)
    assert np.all(indxs == [-1, 1, -1, 2])