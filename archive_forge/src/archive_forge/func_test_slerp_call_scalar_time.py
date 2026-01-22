import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_call_scalar_time():
    r = Rotation.from_euler('X', [0, 80], degrees=True)
    s = Slerp([0, 1], r)
    r_interpolated = s(0.25)
    r_interpolated_expected = Rotation.from_euler('X', 20, degrees=True)
    delta = r_interpolated * r_interpolated_expected.inv()
    assert_allclose(delta.magnitude(), 0, atol=1e-16)