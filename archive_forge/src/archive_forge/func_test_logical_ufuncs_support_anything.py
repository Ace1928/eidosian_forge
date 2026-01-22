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
@pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
def test_logical_ufuncs_support_anything(self, ufunc):
    a = np.array(b'1', dtype='V3')
    c = np.array([1.0, 2.0])
    assert_array_equal(ufunc(a, c), ufunc([True, True], True))
    assert ufunc.reduce(a) == True
    out = np.zeros(2, dtype=np.int32)
    expected = ufunc([True, True], True).astype(out.dtype)
    assert_array_equal(ufunc(a, c, out=out), expected)
    out = np.zeros((), dtype=np.int32)
    assert ufunc.reduce(a, out=out) == True
    a = np.array([3], dtype='i')
    out = np.zeros((), dtype=a.dtype)
    assert ufunc.reduce(a, out=out) == 1