import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_shrunk_covariance():
    """Check consistency between `ShrunkCovariance` and `shrunk_covariance`."""
    cov = ShrunkCovariance(shrinkage=0.5)
    cov.fit(X)
    assert_array_almost_equal(shrunk_covariance(empirical_covariance(X), shrinkage=0.5), cov.covariance_, 4)
    cov = ShrunkCovariance()
    cov.fit(X)
    assert_array_almost_equal(shrunk_covariance(empirical_covariance(X)), cov.covariance_, 4)
    cov = ShrunkCovariance(shrinkage=0.0)
    cov.fit(X)
    assert_array_almost_equal(empirical_covariance(X), cov.covariance_, 4)
    X_1d = X[:, 0].reshape((-1, 1))
    cov = ShrunkCovariance(shrinkage=0.3)
    cov.fit(X_1d)
    assert_array_almost_equal(empirical_covariance(X_1d), cov.covariance_, 4)
    cov = ShrunkCovariance(shrinkage=0.5, store_precision=False)
    cov.fit(X)
    assert cov.precision_ is None