from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('arraylike', arraylikes())
def test_nested_arraylikes(self, arraylike):
    initial = arraylike(np.ones((1, 1)))
    nested = initial
    for i in range(np.MAXDIMS - 1):
        nested = [nested]
    with pytest.raises(ValueError, match='.*would exceed the maximum'):
        np.array(nested, dtype='float64')
    arr = np.array(nested, dtype=object)
    assert arr.shape == (1,) * np.MAXDIMS
    assert arr.item() == np.array(initial).item()