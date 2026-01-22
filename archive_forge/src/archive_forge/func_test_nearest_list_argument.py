import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_nearest_list_argument(self):
    nd = np.array([[0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 2]])
    d = nd[:, 3:]
    NI = NearestNDInterpolator((d[0], d[1]), d[2])
    assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])
    NI = NearestNDInterpolator((d[0], d[1]), list(d[2]))
    assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])