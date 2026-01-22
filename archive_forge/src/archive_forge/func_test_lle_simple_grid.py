from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import (
def test_lle_simple_grid(global_dtype):
    rng = np.random.RandomState(42)
    X = np.array(list(product(range(5), repeat=2)))
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)
    n_components = 2
    clf = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=n_components, random_state=rng)
    tol = 0.1
    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X, 'fro')
    assert reconstruction_error < tol
    for solver in eigen_solvers:
        clf.set_params(eigen_solver=solver)
        clf.fit(X)
        assert clf.embedding_.shape[1] == n_components
        reconstruction_error = linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
        assert reconstruction_error < tol
        assert_allclose(clf.reconstruction_error_, reconstruction_error, atol=0.1)
    noise = rng.randn(*X.shape).astype(global_dtype, copy=False) / 100
    X_reembedded = clf.transform(X + noise)
    assert linalg.norm(X_reembedded - clf.embedding_) < tol