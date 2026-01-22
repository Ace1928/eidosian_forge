from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('arraylike', arraylikes())
def test_uneven_depth_ragged(self, arraylike):
    arr = np.arange(4).reshape((2, 2))
    arr = arraylike(arr)
    out = np.array([arr, [arr]], dtype=object)
    assert out.shape == (2,)
    assert out[0] is arr
    assert type(out[1]) is list
    with pytest.raises(ValueError):
        np.array([arr, [arr, arr]], dtype=object)