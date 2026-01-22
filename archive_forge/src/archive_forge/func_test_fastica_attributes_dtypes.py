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
def test_fastica_attributes_dtypes(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    fica = FastICA(n_components=5, max_iter=1000, whiten='unit-variance', random_state=0).fit(X)
    assert fica.components_.dtype == global_dtype
    assert fica.mixing_.dtype == global_dtype
    assert fica.mean_.dtype == global_dtype
    assert fica.whitening_.dtype == global_dtype