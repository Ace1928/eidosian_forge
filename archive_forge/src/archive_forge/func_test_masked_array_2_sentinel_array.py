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
def test_masked_array_2_sentinel_array():
    np.random.seed(0)
    A = np.random.rand(10, 11, 12)
    B = np.random.rand(12)
    mask = A < 0.5
    A = np.ma.masked_array(A, mask)
    max_float = np.finfo(np.float64).max
    max_float2 = np.nextafter(max_float, -np.inf)
    max_float3 = np.nextafter(max_float2, -np.inf)
    A[3, 4, 1] = np.nan
    A[4, 5, 2] = np.inf
    A[5, 6, 3] = max_float
    B[8] = np.nan
    B[7] = np.inf
    B[6] = max_float2
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([A, B])
    A_out, B_out = out_arrays
    assert sentinel != max_float and sentinel != max_float2
    assert sentinel == max_float3
    A_reference = A.data
    A_reference[A.mask] = sentinel
    np.testing.assert_array_equal(A_out, A_reference)
    assert B_out is B