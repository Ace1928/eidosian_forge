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
def test_as_euler_degenerate_compare_algorithms(seq_tuple, intrinsic):
    angles = np.array([[45, 90, 35], [35, -90, 20], [35, 90, 25], [25, -90, 15]])
    seq = ''.join(seq_tuple)
    if intrinsic:
        seq = seq.upper()
    rot = Rotation.from_euler(seq, angles, degrees=True)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        estimates_quat = rot.as_euler(seq, degrees=True)
    assert_allclose(estimates_matrix[:, [0, 2]], estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)
    assert_allclose(estimates_matrix[:, 1], estimates_quat[:, 1], atol=0, rtol=1e-07)
    angles = np.array([[15, 0, 60], [35, 0, 75], [60, 180, 35], [15, -180, 25]])
    idx = angles[:, 1] == 0
    seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
    if intrinsic:
        seq = seq.upper()
    rot = Rotation.from_euler(seq, angles, degrees=True)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
    with pytest.warns(UserWarning, match='Gimbal lock'):
        estimates_quat = rot.as_euler(seq, degrees=True)
    assert_allclose(estimates_matrix[:, [0, 2]], estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)
    assert_allclose(estimates_matrix[~idx, 1], estimates_quat[~idx, 1], atol=0, rtol=1e-07)
    assert_allclose(estimates_matrix[idx, 1], estimates_quat[idx, 1], atol=1e-06)