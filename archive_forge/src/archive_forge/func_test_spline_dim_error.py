import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('method', ['cubic', 'quintic', 'pchip'])
def test_spline_dim_error(self, method):
    points, values = self._get_sample_4d_4()
    match = 'points in dimension'
    with pytest.raises(ValueError, match=match):
        RegularGridInterpolator(points, values, method=method)
    interp = RegularGridInterpolator(points, values)
    sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match=match):
        interp(sample, method=method)