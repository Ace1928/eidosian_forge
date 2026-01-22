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
@pytest.mark.parametrize('target_feature', range(5))
@pytest.mark.parametrize('est, method', [(LinearRegression(), 'brute'), (GradientBoostingRegressor(random_state=0), 'brute'), (GradientBoostingRegressor(random_state=0), 'recursion'), (HistGradientBoostingRegressor(random_state=0), 'brute'), (HistGradientBoostingRegressor(random_state=0), 'recursion')])
def test_partial_dependence_helpers(est, method, target_feature):
    X, y = make_regression(random_state=0, n_features=5, n_informative=5)
    y = y - y.mean()
    est.fit(X, y)
    features = np.array([target_feature], dtype=np.int32)
    grid = np.array([[0.5], [123]])
    if method == 'brute':
        pdp, predictions = _partial_dependence_brute(est, grid, features, X, response_method='auto')
    else:
        pdp = _partial_dependence_recursion(est, grid, features)
    mean_predictions = []
    for val in (0.5, 123):
        X_ = X.copy()
        X_[:, target_feature] = val
        mean_predictions.append(est.predict(X_).mean())
    pdp = pdp[0]
    rtol = 0.1 if method == 'recursion' else 0.001
    assert np.allclose(pdp, mean_predictions, rtol=rtol)