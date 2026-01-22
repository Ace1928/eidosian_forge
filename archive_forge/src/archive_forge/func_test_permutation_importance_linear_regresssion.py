import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container
def test_permutation_importance_linear_regresssion():
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)
    X = scale(X)
    y = scale(y)
    lr = LinearRegression().fit(X, y)
    expected_importances = 2 * lr.coef_ ** 2
    results = permutation_importance(lr, X, y, n_repeats=50, scoring='neg_mean_squared_error')
    assert_allclose(expected_importances, results.importances_mean, rtol=0.1, atol=1e-06)