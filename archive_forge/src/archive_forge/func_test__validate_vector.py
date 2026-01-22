import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test__validate_vector():
    x = [1, 2, 3]
    y = _validate_vector(x)
    assert_array_equal(y, x)
    y = _validate_vector(x, dtype=np.float64)
    assert_array_equal(y, x)
    assert_equal(y.dtype, np.float64)
    x = [1]
    y = _validate_vector(x)
    assert_equal(y.ndim, 1)
    assert_equal(y, x)
    x = 1
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        _validate_vector(x)
    x = np.arange(5).reshape(1, -1, 1)
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        _validate_vector(x)
    x = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match='Input vector should be 1-D'):
        _validate_vector(x)