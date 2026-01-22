import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_call_time_out_of_range():
    rnd = np.random.RandomState(0)
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    t = np.arange(5) + 1
    s = Slerp(t, r)
    with pytest.raises(ValueError, match='times must be within the range'):
        s([0, 1, 2])
    with pytest.raises(ValueError, match='times must be within the range'):
        s([1, 2, 6])