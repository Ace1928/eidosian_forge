import math
from itertools import product
import numpy as np
import pytest
from scipy.sparse import rand as sparse_rand
from sklearn import clone, datasets, manifold, neighbors, pipeline, preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('n_neighbors, radius', [(24, None), (None, np.inf)])
@pytest.mark.parametrize('eigen_solver', eigen_solvers)
@pytest.mark.parametrize('path_method', path_methods)
def test_isomap_simple_grid(global_dtype, n_neighbors, radius, eigen_solver, path_method):
    n_pts = 25
    X = create_sample_data(global_dtype, n_pts=n_pts, add_noise=False)
    if n_neighbors is not None:
        G = neighbors.kneighbors_graph(X, n_neighbors, mode='distance')
    else:
        G = neighbors.radius_neighbors_graph(X, radius, mode='distance')
    clf = manifold.Isomap(n_neighbors=n_neighbors, radius=radius, n_components=2, eigen_solver=eigen_solver, path_method=path_method)
    clf.fit(X)
    if n_neighbors is not None:
        G_iso = neighbors.kneighbors_graph(clf.embedding_, n_neighbors, mode='distance')
    else:
        G_iso = neighbors.radius_neighbors_graph(clf.embedding_, radius, mode='distance')
    atol = 1e-05 if global_dtype == np.float32 else 0
    assert_allclose_dense_sparse(G, G_iso, atol=atol)