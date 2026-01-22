from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('dtype', [np.int64, np.float32])
@pytest.mark.parametrize('scalar', [param(np.timedelta64(123, 'ns'), id='timedelta64[ns]'), param(np.timedelta64(12, 'generic'), id='timedelta64[generic]')])
def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
    arr = np.array(scalar, dtype=dtype)
    cast = np.array(scalar).astype(dtype)
    ass = np.ones((), dtype=dtype)
    ass[()] = scalar
    assert_array_equal(arr, cast)
    assert_array_equal(cast, cast)