import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_linear_edges(self):
    points, values = self._get_sample_4d()
    interp = RegularGridInterpolator(points, values)
    sample = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    wanted = np.asarray([0.0, 1111.0])
    assert_array_almost_equal(interp(sample), wanted)