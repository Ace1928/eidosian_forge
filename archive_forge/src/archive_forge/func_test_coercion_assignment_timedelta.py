from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize(['val', 'unit'], [param(123, 's', id='[s]'), param(123, 'D', id='[D]')])
def test_coercion_assignment_timedelta(self, val, unit):
    scalar = np.timedelta64(val, unit)
    np.array(scalar, dtype='S6')
    cast = np.array(scalar).astype('S6')
    ass = np.ones((), dtype='S6')
    ass[()] = scalar
    expected = scalar.astype('S')[:6]
    assert cast[()] == expected
    assert ass[()] == expected