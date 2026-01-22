import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_invalid_xi_dimensions(self):
    points = [(0, 1)]
    values = [0, 1]
    xi = np.ones((1, 1, 3))
    msg = 'The requested sample points xi have dimension 3, but this RegularGridInterpolator has dimension 1'
    with assert_raises(ValueError, match=msg):
        interpn(points, values, xi)