import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_xi_1d(self):
    points, values = self._sample_4d_data()
    sample = np.asarray([0.1, 0.1, 10.0, 9.0])
    v1 = interpn(points, values, sample, bounds_error=False)
    v2 = interpn(points, values, sample[None, :], bounds_error=False)
    assert_allclose(v1, v2)