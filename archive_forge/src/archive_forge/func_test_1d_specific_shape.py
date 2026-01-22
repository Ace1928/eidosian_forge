import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
@pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
def test_1d_specific_shape(self, bc):
    lap = LaplacianNd(grid_shape=(6,), boundary_conditions=bc)
    lapa = lap.toarray()
    if bc == 'neumann':
        a = np.array([[-1, 1, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 1, -2, 1, 0, 0], [0, 0, 1, -2, 1, 0], [0, 0, 0, 1, -2, 1], [0, 0, 0, 0, 1, -1]])
    elif bc == 'dirichlet':
        a = np.array([[-2, 1, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0], [0, 1, -2, 1, 0, 0], [0, 0, 1, -2, 1, 0], [0, 0, 0, 1, -2, 1], [0, 0, 0, 0, 1, -2]])
    else:
        a = np.array([[-2, 1, 0, 0, 0, 1], [1, -2, 1, 0, 0, 0], [0, 1, -2, 1, 0, 0], [0, 0, 1, -2, 1, 0], [0, 0, 0, 1, -2, 1], [1, 0, 0, 0, 1, -2]])
    assert_array_equal(a, lapa)