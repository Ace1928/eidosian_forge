import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
@pytest.mark.parametrize('grid_shape', [(6,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
def test_toarray_tosparse_consistency(self, grid_shape, bc):
    lap = LaplacianNd(grid_shape, boundary_conditions=bc)
    n = np.prod(grid_shape)
    assert_array_equal(lap.toarray(), lap(np.eye(n)))
    assert_array_equal(lap.tosparse().toarray(), lap.toarray())