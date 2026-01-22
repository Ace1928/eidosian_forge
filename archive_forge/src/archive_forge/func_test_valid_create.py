import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_valid_create(self):
    points = [(0.0, 0.5, 1.0), (0.0, 1.0, 0.5)]
    values = np.asarray([0.0, 0.5, 1.0])
    values0 = values[:, np.newaxis]
    values1 = values[np.newaxis, :]
    values = values0 + values1 * 10
    assert_raises(ValueError, RegularGridInterpolator, points, values)
    points = [((0.0, 0.5, 1.0),), (0.0, 0.5, 1.0)]
    assert_raises(ValueError, RegularGridInterpolator, points, values)
    points = [(0.0, 0.5, 0.75, 1.0), (0.0, 0.5, 1.0)]
    assert_raises(ValueError, RegularGridInterpolator, points, values)
    points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
    assert_raises(ValueError, RegularGridInterpolator, points, values)
    points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
    assert_raises(ValueError, RegularGridInterpolator, points, values, method='undefmethod')