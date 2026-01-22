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
def test_isomap_fit_precomputed_radius_graph(global_dtype):
    X, y = datasets.make_s_curve(200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    radius = 10
    g = neighbors.radius_neighbors_graph(X, radius=radius, mode='distance')
    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric='precomputed')
    isomap.fit(g)
    precomputed_result = isomap.embedding_
    isomap = manifold.Isomap(n_neighbors=None, radius=radius, metric='minkowski')
    result = isomap.fit_transform(X)
    atol = 1e-05 if global_dtype == np.float32 else 0
    assert_allclose(precomputed_result, result, atol=atol)