import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
@pytest.mark.parametrize('ndim', range(33))
def test_higher_dims(self, ndim):
    shape = (1,) * ndim
    x = np.zeros(shape, dtype=np.float64)
    assert shape == np.from_dlpack(x).shape