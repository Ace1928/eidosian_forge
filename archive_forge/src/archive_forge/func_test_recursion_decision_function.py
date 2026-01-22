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
@pytest.mark.parametrize('est', (GradientBoostingClassifier(random_state=0), HistGradientBoostingClassifier(random_state=0)))
@pytest.mark.parametrize('target_feature', (0, 1, 2, 3, 4, 5))
def test_recursion_decision_function(est, target_feature):
    X, y = make_classification(n_classes=2, n_clusters_per_class=1, random_state=1)
    assert np.mean(y) == 0.5
    est.fit(X, y)
    preds_1 = partial_dependence(est, X, [target_feature], response_method='decision_function', method='recursion', kind='average')
    preds_2 = partial_dependence(est, X, [target_feature], response_method='decision_function', method='brute', kind='average')
    assert_allclose(preds_1['average'], preds_2['average'], atol=1e-07)