import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_oas():
    X_centered = X - X.mean(axis=0)
    oa = OAS(assume_centered=True)
    oa.fit(X_centered)
    shrinkage_ = oa.shrinkage_
    score_ = oa.score(X_centered)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_centered, assume_centered=True)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)
    X_1d = X[:, 0:1]
    oa = OAS(assume_centered=True)
    oa.fit(X_1d)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d, assume_centered=True)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    assert_array_almost_equal((X_1d ** 2).sum() / n_samples, oa.covariance_, 4)
    oa = OAS(store_precision=False, assume_centered=True)
    oa.fit(X_centered)
    assert_almost_equal(oa.score(X_centered), score_, 4)
    assert oa.precision_ is None
    oa = OAS()
    oa.fit(X)
    assert_almost_equal(oa.shrinkage_, shrinkage_, 4)
    assert_almost_equal(oa.score(X), score_, 4)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    scov = ShrunkCovariance(shrinkage=oa.shrinkage_)
    scov.fit(X)
    assert_array_almost_equal(scov.covariance_, oa.covariance_, 4)
    X_1d = X[:, 0].reshape((-1, 1))
    oa = OAS()
    oa.fit(X_1d)
    oa_cov_from_mle, oa_shrinkage_from_mle = oas(X_1d)
    assert_array_almost_equal(oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(oa_shrinkage_from_mle, oa.shrinkage_)
    assert_array_almost_equal(empirical_covariance(X_1d), oa.covariance_, 4)
    X_1sample = np.arange(5).reshape(1, 5)
    oa = OAS()
    warn_msg = 'Only one sample available. You may want to reshape your data array'
    with pytest.warns(UserWarning, match=warn_msg):
        oa.fit(X_1sample)
    assert_array_almost_equal(oa.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))
    oa = OAS(store_precision=False)
    oa.fit(X)
    assert_almost_equal(oa.score(X), score_, 4)
    assert oa.precision_ is None
    X_1f = X[:, 0:1]
    oa = OAS()
    oa.fit(X_1f)
    _oa_cov_from_mle, _oa_shrinkage_from_mle = _oas(X_1f)
    assert_array_almost_equal(_oa_cov_from_mle, oa.covariance_, 4)
    assert_almost_equal(_oa_shrinkage_from_mle, oa.shrinkage_)
    assert_array_almost_equal((X_1f ** 2).sum() / n_samples, oa.covariance_, 4)