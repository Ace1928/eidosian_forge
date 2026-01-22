import numpy as np
from numpy.testing import assert_
import numpy.testing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
def test_minimum_spanning_tree():
    graph = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 8, 5], [0, 0, 8, 0, 1], [0, 0, 5, 1, 0]]
    graph = np.asarray(graph)
    expected = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 5], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    expected = np.asarray(expected)
    csgraph = csr_matrix(graph)
    mintree = minimum_spanning_tree(csgraph)
    mintree_array = mintree.toarray()
    npt.assert_array_equal(mintree_array, expected, 'Incorrect spanning tree found.')
    npt.assert_array_equal(csgraph.toarray(), graph, 'Original graph was modified.')
    mintree = minimum_spanning_tree(csgraph, overwrite=True)
    npt.assert_array_equal(mintree.toarray(), expected, 'Graph was not properly modified to contain MST.')
    np.random.seed(1234)
    for N in (5, 10, 15, 20):
        graph = 3 + np.random.random((N, N))
        csgraph = csr_matrix(graph)
        mintree = minimum_spanning_tree(csgraph)
        assert_(mintree.nnz < N)
        idx = np.arange(N - 1)
        graph[idx, idx + 1] = 1
        csgraph = csr_matrix(graph)
        mintree = minimum_spanning_tree(csgraph)
        expected = np.zeros((N, N))
        expected[idx, idx + 1] = 1
        npt.assert_array_equal(mintree.toarray(), expected, 'Incorrect spanning tree found.')