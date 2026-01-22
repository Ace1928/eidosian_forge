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
def test_fastica_convergence_fail():
    rng = np.random.RandomState(0)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)
    warn_msg = 'FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        ica = FastICA(algorithm='parallel', n_components=2, random_state=rng, max_iter=2, tol=0.0)
        ica.fit(m.T)