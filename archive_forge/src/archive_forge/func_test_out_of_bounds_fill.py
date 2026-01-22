import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_out_of_bounds_fill(self):
    points, values = self._get_sample_4d()
    interp = RegularGridInterpolator(points, values, bounds_error=False, fill_value=np.nan)
    sample = np.asarray([[-0.1, -0.1, -0.1, -0.1], [1.1, 1.1, 1.1, 1.1], [2.1, 2.1, -1.1, -1.1]])
    wanted = np.asarray([np.nan, np.nan, np.nan])
    assert_array_almost_equal(interp(sample, method='nearest'), wanted)
    assert_array_almost_equal(interp(sample, method='linear'), wanted)
    sample = np.asarray([[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]])
    wanted = np.asarray([1001.1, 846.2, 555.5])
    assert_array_almost_equal(interp(sample), wanted)