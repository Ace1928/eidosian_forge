import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('name', NAMES)
def test_single_reduction(name):
    g = Rotation.create_group(name)
    f = g[-1].reduce(g)
    assert_array_almost_equal(f.magnitude(), 0)
    assert f.as_quat().shape == (4,)