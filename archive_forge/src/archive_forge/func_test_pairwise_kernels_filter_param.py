import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
def test_pairwise_kernels_filter_param():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    K = rbf_kernel(X, Y, gamma=0.1)
    params = {'gamma': 0.1, 'blabla': ':)'}
    K2 = pairwise_kernels(X, Y, metric='rbf', filter_params=True, **params)
    assert_allclose(K, K2)
    with pytest.raises(TypeError):
        pairwise_kernels(X, Y, metric='rbf', **params)