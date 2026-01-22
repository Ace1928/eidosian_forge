import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_1d_shape_consistency(self):
    assert_equal(np.ma.median(array([1, 2, 3], mask=[0, 0, 0])).shape, np.ma.median(array([1, 2, 3], mask=[0, 1, 0])).shape)