import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_quat_double_cover():
    q = np.array([0, 0, 0, -1])
    r = Rotation.from_quat(q)
    assert_equal(q, r.as_quat(canonical=False))
    q = np.array([1, 0, 0, 1]) / np.sqrt(2)
    r = Rotation.from_quat(q)
    r3 = r * r * r
    assert_allclose(r.as_quat(canonical=False) * np.sqrt(2), [1, 0, 0, 1])
    assert_allclose(r.inv().as_quat(canonical=False) * np.sqrt(2), [-1, 0, 0, 1])
    assert_allclose(r3.as_quat(canonical=False) * np.sqrt(2), [1, 0, 0, -1])
    assert_allclose(r3.inv().as_quat(canonical=False) * np.sqrt(2), [-1, 0, 0, -1])
    assert_allclose((r * r.inv()).as_quat(canonical=False), [0, 0, 0, 1], atol=2e-16)
    assert_allclose((r3 * r3.inv()).as_quat(canonical=False), [0, 0, 0, 1], atol=2e-16)
    assert_allclose((r * r3).as_quat(canonical=False), [0, 0, 0, -1], atol=2e-16)
    assert_allclose((r.inv() * r3.inv()).as_quat(canonical=False), [0, 0, 0, -1], atol=2e-16)