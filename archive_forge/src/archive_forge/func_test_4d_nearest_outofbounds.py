import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_4d_nearest_outofbounds(self):
    points, values = self._sample_4d_data()
    sample = np.asarray([[0.1, -0.1, 10.1, 9.0]])
    wanted = 999.99
    actual = interpn(points, values, sample, method='nearest', bounds_error=False, fill_value=999.99)
    assert_array_almost_equal(actual, wanted)