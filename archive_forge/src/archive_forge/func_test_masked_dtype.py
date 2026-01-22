from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def test_masked_dtype():
    max16 = np.iinfo(np.int16).max
    max128c = np.finfo(np.complex128).max
    a = np.array([1, 2, max16], dtype=np.int16)
    b = np.ma.array([1, 2, 1], dtype=np.int8, mask=[0, 1, 0])
    c = np.ma.array([1, 2, 1], dtype=np.complex128, mask=[0, 0, 0])
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a, b])
    a_out, b_out = out_arrays
    assert sentinel == max16 - 1
    assert b_out.dtype == np.int16
    assert_allclose(b_out, [b[0], sentinel, b[-1]])
    assert a_out is a
    assert not isinstance(b_out, np.ma.MaskedArray)
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([b, c])
    b_out, c_out = out_arrays
    assert sentinel == max128c
    assert b_out.dtype == np.complex128
    assert_allclose(b_out, [b[0], sentinel, b[-1]])
    assert not isinstance(b_out, np.ma.MaskedArray)
    assert not isinstance(c_out, np.ma.MaskedArray)
    min8, max8 = (np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    a = np.arange(min8, max8 + 1, dtype=np.int8)
    mask1 = np.zeros_like(a, dtype=bool)
    mask0 = np.zeros_like(a, dtype=bool)
    mask1[1] = True
    a1 = np.ma.array(a, mask=mask1)
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a1])
    assert sentinel == min8 + 1
    mask0[0] = True
    a0 = np.ma.array(a, mask=mask0)
    message = 'This function replaces masked elements with sentinel...'
    with pytest.raises(ValueError, match=message):
        _masked_arrays_2_sentinel_arrays([a0])
    a = np.ma.array([1, 2, 3], mask=[0, 1, 0], dtype=np.float32)
    assert stats.gmean(a).dtype == np.float32