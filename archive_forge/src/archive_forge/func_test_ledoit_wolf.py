import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_ledoit_wolf():
    X_centered = X - X.mean(axis=0)
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_centered)
    shrinkage_ = lw.shrinkage_
    score_ = lw.score(X_centered)
    assert_almost_equal(ledoit_wolf_shrinkage(X_centered, assume_centered=True), shrinkage_)
    assert_almost_equal(ledoit_wolf_shrinkage(X_centered, assume_centered=True, block_size=6), shrinkage_)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_centered, assume_centered=True)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_, assume_centered=True)
    scov.fit(X_centered)
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X_1d)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d, assume_centered=True)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    assert_array_almost_equal((X_1d ** 2).sum() / n_samples, lw.covariance_, 4)
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(X_centered)
    assert_almost_equal(lw.score(X_centered), score_, 4)
    assert lw.precision_ is None
    lw = LedoitWolf()
    lw.fit(X)
    assert_almost_equal(lw.shrinkage_, shrinkage_, 4)
    assert_almost_equal(lw.shrinkage_, ledoit_wolf_shrinkage(X))
    assert_almost_equal(lw.shrinkage_, ledoit_wolf(X)[1])
    assert_almost_equal(lw.shrinkage_, _ledoit_wolf(X=X, assume_centered=False, block_size=10000)[1])
    assert_almost_equal(lw.score(X), score_, 4)
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    scov = ShrunkCovariance(shrinkage=lw.shrinkage_)
    scov.fit(X)
    assert_array_almost_equal(scov.covariance_, lw.covariance_, 4)
    X_1d = X[:, 0].reshape((-1, 1))
    lw = LedoitWolf()
    lw.fit(X_1d)
    assert_allclose(X_1d.var(ddof=0), _ledoit_wolf(X=X_1d, assume_centered=False, block_size=10000)[0])
    lw_cov_from_mle, lw_shrinkage_from_mle = ledoit_wolf(X_1d)
    assert_array_almost_equal(lw_cov_from_mle, lw.covariance_, 4)
    assert_almost_equal(lw_shrinkage_from_mle, lw.shrinkage_)
    assert_array_almost_equal(empirical_covariance(X_1d), lw.covariance_, 4)
    X_1sample = np.arange(5).reshape(1, 5)
    lw = LedoitWolf()
    warn_msg = 'Only one sample available. You may want to reshape your data array'
    with pytest.warns(UserWarning, match=warn_msg):
        lw.fit(X_1sample)
    assert_array_almost_equal(lw.covariance_, np.zeros(shape=(5, 5), dtype=np.float64))
    lw = LedoitWolf(store_precision=False)
    lw.fit(X)
    assert_almost_equal(lw.score(X), score_, 4)
    assert lw.precision_ is None