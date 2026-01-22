import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_array_almost_equal
def test_feature_agglomeration():
    n_clusters = 1
    X = np.array([0, 0, 1]).reshape(1, 3)
    agglo_mean = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.mean)
    agglo_median = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.median)
    agglo_mean.fit(X)
    agglo_median.fit(X)
    assert np.size(np.unique(agglo_mean.labels_)) == n_clusters
    assert np.size(np.unique(agglo_median.labels_)) == n_clusters
    assert np.size(agglo_mean.labels_) == X.shape[1]
    assert np.size(agglo_median.labels_) == X.shape[1]
    Xt_mean = agglo_mean.transform(X)
    Xt_median = agglo_median.transform(X)
    assert Xt_mean.shape[1] == n_clusters
    assert Xt_median.shape[1] == n_clusters
    assert Xt_mean == np.array([1 / 3.0])
    assert Xt_median == np.array([0.0])
    X_full_mean = agglo_mean.inverse_transform(Xt_mean)
    X_full_median = agglo_median.inverse_transform(Xt_median)
    assert np.unique(X_full_mean[0]).size == n_clusters
    assert np.unique(X_full_median[0]).size == n_clusters
    assert_array_almost_equal(agglo_mean.transform(X_full_mean), Xt_mean)
    assert_array_almost_equal(agglo_median.transform(X_full_median), Xt_median)