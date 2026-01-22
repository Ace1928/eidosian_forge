import numpy as np
import pytest
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
def test_mutual_reachability_graph_equivalence_dense_sparse():
    """Check that we get the same results for dense and sparse implementation."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 5)
    X_dense = X.T @ X
    X_sparse = _convert_container(X_dense, 'sparse_csr')
    mr_graph_dense = mutual_reachability_graph(X_dense, min_samples=3)
    mr_graph_sparse = mutual_reachability_graph(X_sparse, min_samples=3)
    assert_allclose(mr_graph_dense, mr_graph_sparse.toarray())