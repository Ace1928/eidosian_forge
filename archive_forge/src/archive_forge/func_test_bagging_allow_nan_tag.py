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
@pytest.mark.parametrize('bagging, expected_allow_nan', [(BaggingClassifier(HistGradientBoostingClassifier(max_iter=1)), True), (BaggingRegressor(HistGradientBoostingRegressor(max_iter=1)), True), (BaggingClassifier(LogisticRegression()), False), (BaggingRegressor(SVR()), False)])
def test_bagging_allow_nan_tag(bagging, expected_allow_nan):
    """Check that bagging inherits allow_nan tag."""
    assert bagging._get_tags()['allow_nan'] == expected_allow_nan