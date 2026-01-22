import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_testAverage4(self):
    x = np.array([2, 3, 4]).reshape(3, 1)
    b = np.ma.array(x, mask=[[False], [False], [True]])
    w = np.array([4, 5, 6]).reshape(3, 1)
    actual = average(b, weights=w, axis=1, keepdims=True)
    desired = masked_array([[2.0], [3.0], [4.0]], [[False], [False], [True]])
    assert_equal(actual, desired)