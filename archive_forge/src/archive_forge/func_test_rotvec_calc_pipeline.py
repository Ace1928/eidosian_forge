import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_rotvec_calc_pipeline():
    rotvec = np.array([[0, 0, 0], [1, -1, 2], [-0.0003, 0.00035, 7.5e-05]])
    assert_allclose(Rotation.from_rotvec(rotvec).as_rotvec(), rotvec)
    assert_allclose(Rotation.from_rotvec(rotvec, degrees=True).as_rotvec(degrees=True), rotvec)