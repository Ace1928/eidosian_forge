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
def test_isomap_clone_bug():
    model = manifold.Isomap()
    for n_neighbors in [10, 15, 20]:
        model.set_params(n_neighbors=n_neighbors)
        model.fit(np.random.rand(50, 2))
        assert model.nbrs_.n_neighbors == n_neighbors