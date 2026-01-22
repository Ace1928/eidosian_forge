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
@pytest.mark.parametrize('n_neighbors, radius', [(2, None), (None, 10.0)])
def test_pipeline(n_neighbors, radius, global_dtype):
    X, y = datasets.make_blobs(random_state=0)
    X = X.astype(global_dtype, copy=False)
    clf = pipeline.Pipeline([('isomap', manifold.Isomap(n_neighbors=n_neighbors, radius=radius)), ('clf', neighbors.KNeighborsClassifier())])
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)