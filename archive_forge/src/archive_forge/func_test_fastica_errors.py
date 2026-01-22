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
def test_fastica_errors():
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    w_init = rng.randn(n_features + 1, n_features + 1)
    with pytest.raises(ValueError, match='alpha must be in \\[1,2\\]'):
        fastica(X, fun_args={'alpha': 0})
    with pytest.raises(ValueError, match='w_init has invalid shape.+should be \\(3L?, 3L?\\)'):
        fastica(X, w_init=w_init)