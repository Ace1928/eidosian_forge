import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('seed', range(3))
def test_poisson_zero_nodes(seed):
    X = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 2], [1, 3]]
    y = [0, 0, 0, 0, 1, 2, 3, 4]
    reg = DecisionTreeRegressor(criterion='squared_error', random_state=seed)
    reg.fit(X, y)
    assert np.amin(reg.predict(X)) == 0
    reg = DecisionTreeRegressor(criterion='poisson', random_state=seed)
    reg.fit(X, y)
    assert np.all(reg.predict(X) > 0)
    n_features = 10
    X, y = datasets.make_regression(effective_rank=n_features * 2 // 3, tail_strength=0.6, n_samples=1000, n_features=n_features, n_informative=n_features * 2 // 3, random_state=seed)
    y[(-1 < y) & (y < 0)] = 0
    y = np.abs(y)
    reg = DecisionTreeRegressor(criterion='poisson', random_state=seed)
    reg.fit(X, y)
    assert np.all(reg.predict(X) > 0)