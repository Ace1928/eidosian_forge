import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('n', NS)
@pytest.mark.parametrize('axis', 'XYZ')
def test_dicyclic(n, axis):
    """Test that the dicyclic group correctly fixes the rotations of a
    prism."""
    P = _generate_prism(n, axis='XYZ'.index(axis))
    for g in Rotation.create_group('D%d' % n, axis=axis):
        assert _calculate_rmsd(P, g.apply(P)) < TOL