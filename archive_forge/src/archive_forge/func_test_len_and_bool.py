import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_len_and_bool():
    rotation_multi_empty = Rotation(np.empty((0, 4)))
    rotation_multi_one = Rotation([[0, 0, 0, 1]])
    rotation_multi = Rotation([[0, 0, 0, 1], [0, 0, 0, 1]])
    rotation_single = Rotation([0, 0, 0, 1])
    assert len(rotation_multi_empty) == 0
    assert len(rotation_multi_one) == 1
    assert len(rotation_multi) == 2
    with pytest.raises(TypeError, match='Single rotation has no len().'):
        len(rotation_single)
    assert rotation_multi_empty
    assert rotation_multi_one
    assert rotation_multi
    assert rotation_single