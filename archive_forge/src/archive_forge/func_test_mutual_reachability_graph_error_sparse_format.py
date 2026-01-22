import numpy as np
import pytest
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
def test_mutual_reachability_graph_error_sparse_format():
    """Check that we raise an error if the sparse format is not CSR."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 10)
    X = X.T @ X
    np.fill_diagonal(X, 0.0)
    X = _convert_container(X, 'sparse_csc')
    err_msg = 'Only sparse CSR matrices are supported'
    with pytest.raises(ValueError, match=err_msg):
        mutual_reachability_graph(X)