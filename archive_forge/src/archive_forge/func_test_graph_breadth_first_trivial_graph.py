import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_array
from scipy.sparse.csgraph import (breadth_first_tree, depth_first_tree,
def test_graph_breadth_first_trivial_graph():
    csgraph = np.array([[0]])
    csgraph = csgraph_from_dense(csgraph, null_value=0)
    bfirst = np.array([[0]])
    for directed in [True, False]:
        bfirst_test = breadth_first_tree(csgraph, 0, directed)
        assert_array_almost_equal(csgraph_to_dense(bfirst_test), bfirst)