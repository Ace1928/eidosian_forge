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
@pytest.mark.parametrize('list_single_scorer, multi_scorer', [(['r2', 'neg_mean_squared_error'], ['r2', 'neg_mean_squared_error']), (['r2', 'neg_mean_squared_error'], {'r2': get_scorer('r2'), 'neg_mean_squared_error': get_scorer('neg_mean_squared_error')}), (['r2', 'neg_mean_squared_error'], lambda estimator, X, y: {'r2': r2_score(y, estimator.predict(X)), 'neg_mean_squared_error': -mean_squared_error(y, estimator.predict(X))})])
def test_permutation_importance_multi_metric(list_single_scorer, multi_scorer):
    x, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(x, y)
    multi_importance = permutation_importance(lr, x, y, random_state=1, scoring=multi_scorer, n_repeats=2)
    assert set(multi_importance.keys()) == set(list_single_scorer)
    for scorer in list_single_scorer:
        multi_result = multi_importance[scorer]
        single_result = permutation_importance(lr, x, y, random_state=1, scoring=scorer, n_repeats=2)
        assert_allclose(multi_result.importances, single_result.importances)