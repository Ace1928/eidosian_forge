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
def test_set_oob_score_label_encoding():
    random_state = 5
    X = [[-1], [0], [1]] * 5
    Y1 = ['A', 'B', 'C'] * 5
    Y2 = [-1, 0, 1] * 5
    Y3 = [0, 1, 2] * 5
    x1 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y1).oob_score_
    x2 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y2).oob_score_
    x3 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y3).oob_score_
    assert [x1, x2] == [x3, x3]