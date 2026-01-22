import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('seed', range(10))
def test_lda_shrinkage(seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=100)
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage=0.5, solver='lsqr')
    c2 = LinearDiscriminantAnalysis(store_covariance=True, covariance_estimator=ShrunkCovariance(shrinkage=0.5), solver='lsqr')
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)