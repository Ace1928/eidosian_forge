import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('TreeClassifier', TREE_BASED_CLASSIFIER_CLASSES)
@pytest.mark.parametrize('depth_first_builder', (True, False))
@pytest.mark.parametrize('sparse_splitter', (True, False))
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_monotonic_constraints_classifications(TreeClassifier, depth_first_builder, sparse_splitter, global_random_seed, csc_container):
    n_samples = 1000
    n_samples_train = 900
    X, y = make_classification(n_samples=n_samples, n_classes=2, n_features=5, n_informative=5, n_redundant=0, random_state=global_random_seed)
    X_train, y_train = (X[:n_samples_train], y[:n_samples_train])
    X_test, _ = (X[n_samples_train:], y[n_samples_train:])
    X_test_0incr, X_test_0decr = (np.copy(X_test), np.copy(X_test))
    X_test_1incr, X_test_1decr = (np.copy(X_test), np.copy(X_test))
    X_test_0incr[:, 0] += 10
    X_test_0decr[:, 0] -= 10
    X_test_1incr[:, 1] += 10
    X_test_1decr[:, 1] -= 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1
    if depth_first_builder:
        est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst)
    else:
        est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst, max_leaf_nodes=n_samples_train)
    if hasattr(est, 'random_state'):
        est.set_params(**{'random_state': global_random_seed})
    if hasattr(est, 'n_estimators'):
        est.set_params(**{'n_estimators': 5})
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    proba_test = est.predict_proba(X_test)
    assert np.logical_and(proba_test >= 0.0, proba_test <= 1.0).all(), 'Probability should always be in [0, 1] range.'
    assert_allclose(proba_test.sum(axis=1), 1.0)
    assert np.all(est.predict_proba(X_test_0incr)[:, 1] >= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_0decr)[:, 1] <= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_1incr)[:, 1] <= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_1decr)[:, 1] >= proba_test[:, 1])