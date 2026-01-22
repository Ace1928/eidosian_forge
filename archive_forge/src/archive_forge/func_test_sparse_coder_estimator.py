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
def test_sparse_coder_estimator():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    coder = SparseCoder(dictionary=V, transform_algorithm='lasso_lars', transform_alpha=0.001).transform(X)
    assert not np.all(coder == 0)
    assert np.sqrt(np.sum((np.dot(coder, V) - X) ** 2)) < 0.1