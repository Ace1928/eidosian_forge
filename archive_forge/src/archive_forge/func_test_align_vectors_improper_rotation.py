import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_improper_rotation():
    x = np.array([[0.89299824, -0.44372674, 0.0752378], [0.60221789, -0.47564102, -0.6411702]])
    y = np.array([[0.02386536, -0.82176463, 0.5693271], [-0.27654929, -0.95191427, -0.1318321]])
    est, rssd = Rotation.align_vectors(x, y)
    assert_allclose(x, est.apply(y), atol=1e-06)
    assert_allclose(rssd, 0, atol=1e-07)