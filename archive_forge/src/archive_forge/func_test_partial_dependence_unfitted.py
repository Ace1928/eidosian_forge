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
@pytest.mark.parametrize('estimator', [LinearRegression(), LogisticRegression(), GradientBoostingRegressor(), GradientBoostingClassifier()])
def test_partial_dependence_unfitted(estimator):
    X = iris.data
    preprocessor = make_column_transformer((StandardScaler(), [0, 2]), (RobustScaler(), [1, 3]))
    pipe = make_pipeline(preprocessor, estimator)
    with pytest.raises(NotFittedError, match='is not fitted yet'):
        partial_dependence(pipe, X, features=[0, 2], grid_resolution=10)
    with pytest.raises(NotFittedError, match='is not fitted yet'):
        partial_dependence(estimator, X, features=[0, 2], grid_resolution=10)