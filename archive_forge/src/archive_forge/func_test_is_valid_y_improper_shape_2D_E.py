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
def test_is_valid_y_improper_shape_2D_E(self):
    y = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        is_valid_y_throw(y)