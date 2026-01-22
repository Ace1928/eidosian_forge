import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_rssd_sensitivity():
    rssd_expected = 0.141421356237308
    sens_expected = np.array([[0.2, 0.0, 0.0], [0.0, 1.5, 1.0], [0.0, 1.0, 1.0]])
    atol = 1e-06
    a = [[0, 1, 0], [0, 1, 1], [0, 1, 1]]
    b = [[1, 0, 0], [1, 1.1, 0], [1, 0.9, 0]]
    rot, rssd, sens = Rotation.align_vectors(a, b, return_sensitivity=True)
    assert np.isclose(rssd, rssd_expected, atol=atol)
    assert np.allclose(sens, sens_expected, atol=atol)