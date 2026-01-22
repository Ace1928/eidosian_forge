import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_magnitude_single_rotation():
    r = Rotation.from_quat(np.eye(4))
    result1 = r[0].magnitude()
    assert_allclose(result1, np.pi)
    result2 = r[3].magnitude()
    assert_allclose(result2, 0)