import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_xi_nd(self):
    points, values = self._sample_4d_data()
    np.random.seed(1234)
    sample = np.random.rand(2, 3, 4)
    v1 = interpn(points, values, sample, method='nearest', bounds_error=False)
    assert_equal(v1.shape, (2, 3))
    v2 = interpn(points, values, sample.reshape(-1, 4), method='nearest', bounds_error=False)
    assert_allclose(v1, v2.reshape(v1.shape))