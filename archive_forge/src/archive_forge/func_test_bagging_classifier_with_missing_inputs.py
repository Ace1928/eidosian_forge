from itertools import cycle, product
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_bagging_classifier_with_missing_inputs():
    X = np.array([[1, 3, 5], [2, None, 6], [2, np.nan, 6], [2, np.inf, 6], [2, -np.inf, 6]])
    y = np.array([3, 6, 6, 6, 6])
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(FunctionTransformer(replace), classifier)
    pipeline.fit(X, y).predict(X)
    bagging_classifier = BaggingClassifier(pipeline)
    bagging_classifier.fit(X, y)
    y_hat = bagging_classifier.predict(X)
    assert y.shape == y_hat.shape
    bagging_classifier.predict_log_proba(X)
    bagging_classifier.predict_proba(X)
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(classifier)
    with pytest.raises(ValueError):
        pipeline.fit(X, y)
    bagging_classifier = BaggingClassifier(pipeline)
    with pytest.raises(ValueError):
        bagging_classifier.fit(X, y)