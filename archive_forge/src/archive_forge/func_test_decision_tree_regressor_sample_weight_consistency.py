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
@pytest.mark.parametrize('criterion', REG_CRITERIONS)
def test_decision_tree_regressor_sample_weight_consistency(criterion):
    """Test that the impact of sample_weight is consistent."""
    tree_params = dict(criterion=criterion)
    tree = DecisionTreeRegressor(**tree_params, random_state=42)
    for kind in ['zeros', 'ones']:
        check_sample_weights_invariance('DecisionTreeRegressor_' + criterion, tree, kind='zeros')
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = np.mean(X, axis=1) + rng.rand(n_samples)
    y += np.min(y) + 0.1
    X2 = np.concatenate([X, X[:n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[:n_samples // 2]])
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[:n_samples // 2] = 2
    tree1 = DecisionTreeRegressor(**tree_params).fit(X, y, sample_weight=sample_weight_1)
    tree2 = DecisionTreeRegressor(**tree_params).fit(X2, y2, sample_weight=None)
    assert tree1.tree_.node_count == tree2.tree_.node_count
    assert_allclose(tree1.predict(X), tree2.predict(X))