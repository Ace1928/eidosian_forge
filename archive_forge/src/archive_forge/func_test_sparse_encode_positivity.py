import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
@pytest.mark.parametrize('algo', ['lasso_lars', 'lasso_cd', 'threshold'])
@pytest.mark.parametrize('positive', [False, True])
def test_sparse_encode_positivity(algo, positive):
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    code = sparse_encode(X, V, algorithm=algo, positive=positive)
    if positive:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()