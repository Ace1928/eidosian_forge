import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_parallel():
    atol = 1e-12
    a = [[1, 0, 0], [0, 1, 0]]
    b = [[0, 1, 0], [0, 1, 0]]
    m_expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    R, _ = Rotation.align_vectors(a[0], b[0])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    assert_allclose(R.apply(b[0]), a[0], atol=atol)
    b = [[1, 0, 0], [1, 0, 0]]
    m_expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    R, _ = Rotation.align_vectors(a[0], b[0])
    assert_allclose(R.as_matrix(), m_expected, atol=atol)
    assert_allclose(R.apply(b[0]), a[0], atol=atol)