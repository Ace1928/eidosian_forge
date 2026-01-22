from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('obj', [object(), 1.2, 10 ** 43, None, 'string'], ids=['object', '1.2', '10**43', 'None', 'string'])
def test_nested_arrays_stringlength(self, obj):
    length = len(str(obj))
    expected = np.dtype(f'S{length}')
    arr = np.array(obj, dtype='O')
    assert np.array([arr, arr], dtype='S').dtype == expected