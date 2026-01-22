import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from scipy.sparse import csgraph, csr_array
def test_weak_connections():
    Xde = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    Xsp = csgraph.csgraph_from_dense(Xde, null_value=0)
    for X in (Xsp, Xde):
        n_components, labels = csgraph.connected_components(X, directed=True, connection='weak')
        assert_equal(n_components, 2)
        assert_array_almost_equal(labels, [0, 0, 1])