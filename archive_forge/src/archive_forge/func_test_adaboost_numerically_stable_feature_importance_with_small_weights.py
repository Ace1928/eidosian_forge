import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_adaboost_numerically_stable_feature_importance_with_small_weights():
    """Check that we don't create NaN feature importance with numerically
    instable inputs.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20320
    """
    rng = np.random.RandomState(42)
    X = rng.normal(size=(1000, 10))
    y = rng.choice([0, 1], size=1000)
    sample_weight = np.ones_like(y) * 1e-263
    tree = DecisionTreeClassifier(max_depth=10, random_state=12)
    ada_model = AdaBoostClassifier(estimator=tree, n_estimators=20, algorithm='SAMME', random_state=12)
    ada_model.fit(X, y, sample_weight=sample_weight)
    assert np.isnan(ada_model.feature_importances_).sum() == 0