import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('seed', range(1))
def test_recursion_decision_tree_vs_forest_and_gbdt(seed):
    rng = np.random.RandomState(seed)
    n_samples = 1000
    n_features = 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples) * 10
    y = y - y.mean()
    max_depth = 5
    tree_seed = 0
    forest = RandomForestRegressor(n_estimators=1, max_features=None, bootstrap=False, max_depth=max_depth, random_state=tree_seed)
    equiv_random_state = check_random_state(tree_seed).randint(np.iinfo(np.int32).max)
    gbdt = GradientBoostingRegressor(n_estimators=1, learning_rate=1, criterion='squared_error', max_depth=max_depth, random_state=equiv_random_state)
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=equiv_random_state)
    forest.fit(X, y)
    gbdt.fit(X, y)
    tree.fit(X, y)
    try:
        assert_is_subtree(tree.tree_, gbdt[0, 0].tree_)
        assert_is_subtree(tree.tree_, forest[0].tree_)
    except AssertionError:
        assert _IS_32BIT, 'this should only fail on 32 bit platforms'
        return
    grid = rng.randn(50).reshape(-1, 1)
    for f in range(n_features):
        features = np.array([f], dtype=np.int32)
        pdp_forest = _partial_dependence_recursion(forest, grid, features)
        pdp_gbdt = _partial_dependence_recursion(gbdt, grid, features)
        pdp_tree = _partial_dependence_recursion(tree, grid, features)
        np.testing.assert_allclose(pdp_gbdt, pdp_tree)
        np.testing.assert_allclose(pdp_forest, pdp_tree)