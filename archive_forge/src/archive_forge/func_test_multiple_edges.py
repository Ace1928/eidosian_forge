import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense
def test_multiple_edges():
    np.random.seed(1234)
    X = np.random.random((10, 10))
    Xcsr = csr_matrix(X)
    Xcsr.indices[::2] = Xcsr.indices[1::2]
    Xdense = Xcsr.toarray()
    assert_array_almost_equal(Xdense[:, 1::2], X[:, ::2] + X[:, 1::2])
    Xdense = csgraph_to_dense(Xcsr)
    assert_array_almost_equal(Xdense[:, 1::2], np.minimum(X[:, ::2], X[:, 1::2]))