import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_davenport_degenerate():
    rnd = np.random.RandomState(0)
    n = 5
    angles = np.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles_middle = [rnd.choice([0, np.pi]) for i in range(n)]
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    lambdas = rnd.uniform(low=0, high=np.pi, size=(5,))
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    for lamb in lambdas:
        ax_lamb = [e1, e2, Rotation.from_rotvec(lamb * e2).apply(e1)]
        angles[:, 1] = angles_middle - lamb
        for order in ['extrinsic', 'intrinsic']:
            ax = ax_lamb if order == 'intrinsic' else ax_lamb[::-1]
            rot = Rotation.from_davenport(ax, order, angles)
            with pytest.warns(UserWarning, match='Gimbal lock'):
                angles_dav = rot.as_davenport(ax, order)
            mat_expected = rot.as_matrix()
            mat_estimated = Rotation.from_davenport(ax, order, angles_dav).as_matrix()
            assert_array_almost_equal(mat_expected, mat_estimated)