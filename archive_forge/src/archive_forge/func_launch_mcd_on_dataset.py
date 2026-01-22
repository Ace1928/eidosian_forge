import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def launch_mcd_on_dataset(n_samples, n_features, n_outliers, tol_loc, tol_cov, tol_support, seed):
    rand_gen = np.random.RandomState(seed)
    data = rand_gen.randn(n_samples, n_features)
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10.0 * (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False
    pure_data = data[inliers_mask]
    mcd_fit = MinCovDet(random_state=seed).fit(data)
    T = mcd_fit.location_
    S = mcd_fit.covariance_
    H = mcd_fit.support_
    error_location = np.mean((pure_data.mean(0) - T) ** 2)
    assert error_location < tol_loc
    error_cov = np.mean((empirical_covariance(pure_data) - S) ** 2)
    assert error_cov < tol_cov
    assert np.sum(H) >= tol_support
    assert_array_almost_equal(mcd_fit.mahalanobis(data), mcd_fit.dist_)