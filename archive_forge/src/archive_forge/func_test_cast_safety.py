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
@pytest.mark.parametrize('ufunc', [np.add, np.sqrt])
def test_cast_safety(self, ufunc):
    """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """

    def call_ufunc(arr, **kwargs):
        return ufunc(*(arr,) * ufunc.nin, **kwargs)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr_bs = arr.astype(arr.dtype.newbyteorder())
    expected = call_ufunc(arr)
    res = call_ufunc(arr, casting='no')
    assert_array_equal(expected, res)
    with pytest.raises(TypeError):
        call_ufunc(arr_bs, casting='no')
    res = call_ufunc(arr_bs, casting='equiv')
    assert_array_equal(expected, res)
    with pytest.raises(TypeError):
        call_ufunc(arr_bs, dtype=np.float64, casting='equiv')
    res = call_ufunc(arr_bs, dtype=np.float64, casting='safe')
    expected = call_ufunc(arr.astype(np.float64))
    assert_array_equal(expected, res)