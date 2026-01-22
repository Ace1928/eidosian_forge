import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_identity_magnitude():
    n = 10
    assert_allclose(Rotation.identity(n).magnitude(), 0)
    assert_allclose(Rotation.identity(n).inv().magnitude(), 0)