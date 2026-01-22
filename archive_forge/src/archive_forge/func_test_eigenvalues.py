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
def test_eigenvalues(self, grid_shape, bc):
    lap = LaplacianNd(grid_shape, boundary_conditions=bc, dtype=np.float64)
    L = lap.toarray()
    eigvals = eigh(L, eigvals_only=True)
    n = np.prod(grid_shape)
    eigenvalues = lap.eigenvalues()
    dtype = eigenvalues.dtype
    atol = n * n * np.finfo(dtype).eps
    assert_allclose(eigenvalues, eigvals, atol=atol)
    for m in np.arange(1, n + 1):
        assert_array_equal(lap.eigenvalues(m), eigenvalues[-m:])