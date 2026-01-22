import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def test_unsupervised_radius_neighbors(global_dtype, n_samples=20, n_features=5, n_query_pts=2, radius=0.5, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    for p in P:
        results = []
        for algorithm in ALGORITHMS:
            neigh = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm, p=p)
            neigh.fit(X)
            ind1 = neigh.radius_neighbors(test, return_distance=False)
            dist, ind = neigh.radius_neighbors(test, return_distance=True)
            for d, i, i1 in zip(dist, ind, ind1):
                j = d.argsort()
                d[:] = d[j]
                i[:] = i[j]
                i1[:] = i1[j]
            results.append((dist, ind))
            assert_allclose(np.concatenate(list(ind)), np.concatenate(list(ind1)))
        for i in range(len(results) - 1):
            (assert_allclose(np.concatenate(list(results[i][0])), np.concatenate(list(results[i + 1][0]))),)
            assert_allclose(np.concatenate(list(results[i][1])), np.concatenate(list(results[i + 1][1])))