import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
def test_gs():
    rng = np.random.RandomState(0)
    W, _, _ = np.linalg.svd(rng.randn(10, 10))
    w = rng.randn(10)
    _gs_decorrelation(w, W, 10)
    assert (w ** 2).sum() < 1e-10
    w = rng.randn(10)
    u = _gs_decorrelation(w, W, 5)
    tmp = np.dot(u, W.T)
    assert (tmp[:5] ** 2).sum() < 1e-10