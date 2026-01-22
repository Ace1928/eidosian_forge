import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@parametrize_rgi_interp_methods
def test_nonscalar_values(self, method):
    points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [(0.0, 5.0, 10.0, 15.0, 20, 25.0)] * 2
    rng = np.random.default_rng(1234)
    values = rng.random((6, 6, 6, 6, 8))
    sample = rng.random((7, 3, 4))
    v = interpn(points, values, sample, method=method, bounds_error=False)
    assert_equal(v.shape, (7, 3, 8), err_msg=method)
    vs = [interpn(points, values[..., j], sample, method=method, bounds_error=False) for j in range(8)]
    v2 = np.array(vs).transpose(1, 2, 0)
    assert_allclose(v, v2, atol=1e-14, err_msg=method)