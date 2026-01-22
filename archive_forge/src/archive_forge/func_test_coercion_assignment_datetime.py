from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('dtype', ['S6', 'U6'])
@pytest.mark.parametrize(['val', 'unit'], [param(123, 's', id='[s]'), param(123, 'D', id='[D]')])
def test_coercion_assignment_datetime(self, val, unit, dtype):
    scalar = np.datetime64(val, unit)
    dtype = np.dtype(dtype)
    cut_string = dtype.type(str(scalar)[:6])
    arr = np.array(scalar, dtype=dtype)
    assert arr[()] == cut_string
    ass = np.ones((), dtype=dtype)
    ass[()] = scalar
    assert ass[()] == cut_string
    with pytest.raises(RuntimeError):
        np.array(scalar).astype(dtype)