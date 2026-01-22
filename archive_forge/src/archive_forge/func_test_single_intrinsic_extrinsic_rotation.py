import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_single_intrinsic_extrinsic_rotation():
    extrinsic = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    intrinsic = Rotation.from_euler('Z', 90, degrees=True).as_matrix()
    assert_allclose(extrinsic, intrinsic)