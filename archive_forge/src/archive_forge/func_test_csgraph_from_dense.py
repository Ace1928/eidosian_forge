import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense
def test_csgraph_from_dense():
    np.random.seed(1234)
    G = np.random.random((10, 10))
    some_nulls = G < 0.4
    all_nulls = G < 0.8
    for null_value in [0, np.nan, np.inf]:
        G[all_nulls] = null_value
        with np.errstate(invalid='ignore'):
            G_csr = csgraph_from_dense(G, null_value=0)
        G[all_nulls] = 0
        assert_array_almost_equal(G, G_csr.toarray())
    for null_value in [np.nan, np.inf]:
        G[all_nulls] = 0
        G[some_nulls] = null_value
        with np.errstate(invalid='ignore'):
            G_csr = csgraph_from_dense(G, null_value=0)
        G[all_nulls] = 0
        assert_array_almost_equal(G, G_csr.toarray())