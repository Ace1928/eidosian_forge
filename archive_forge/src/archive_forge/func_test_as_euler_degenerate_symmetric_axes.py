import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
@pytest.mark.parametrize('seq_tuple', permutations('xyz'))
@pytest.mark.parametrize('intrinsic', (False, True))
def test_as_euler_degenerate_symmetric_axes(seq_tuple, intrinsic):
    angles = np.array([[15, 0, 60], [35, 0, 75], [60, 180, 35], [15, -180, 25]])
    seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        seq = seq.upper()
    rotation = Rotation.from_euler(seq, angles, degrees=True)
    mat_expected = rotation.as_matrix()
    with pytest.warns(UserWarning, match='Gimbal lock'):
        angle_estimates = rotation.as_euler(seq, degrees=True)
    mat_estimated = Rotation.from_euler(seq, angle_estimates, degrees=True).as_matrix()
    assert_array_almost_equal(mat_expected, mat_estimated)