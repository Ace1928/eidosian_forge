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
def test_qda_prior_copy():
    """Check that altering `priors` without `fit` doesn't change `priors_`"""
    priors = np.array([0.5, 0.5])
    qda = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)
    assert_array_equal(qda.priors_, qda.priors)
    priors[0] = 0.2
    assert qda.priors_[0] != qda.priors[0]