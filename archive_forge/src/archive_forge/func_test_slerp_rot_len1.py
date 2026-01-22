import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_rot_len1():
    msg = 'must be a sequence of at least 2 rotations'
    with pytest.raises(ValueError, match=msg):
        r = Rotation.from_quat([[1, 2, 3, 4]])
        Slerp([1], r)