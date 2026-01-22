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
@pytest.mark.parametrize('Estimator, method, data', [(GradientBoostingClassifier, 'auto', binary_classification_data), (GradientBoostingClassifier, 'auto', multiclass_classification_data), (GradientBoostingClassifier, 'brute', binary_classification_data), (GradientBoostingClassifier, 'brute', multiclass_classification_data), (GradientBoostingRegressor, 'auto', regression_data), (GradientBoostingRegressor, 'brute', regression_data), (DecisionTreeRegressor, 'brute', regression_data), (LinearRegression, 'brute', regression_data), (LinearRegression, 'brute', multioutput_regression_data), (LogisticRegression, 'brute', binary_classification_data), (LogisticRegression, 'brute', multiclass_classification_data), (MultiTaskLasso, 'brute', multioutput_regression_data)])
@pytest.mark.parametrize('grid_resolution', (5, 10))
@pytest.mark.parametrize('features', ([1], [1, 2]))
@pytest.mark.parametrize('kind', ('average', 'individual', 'both'))
def test_output_shape(Estimator, method, data, grid_resolution, features, kind):
    est = Estimator()
    if hasattr(est, 'n_estimators'):
        est.set_params(n_estimators=2)
    (X, y), n_targets = data
    n_instances = X.shape[0]
    est.fit(X, y)
    result = partial_dependence(est, X=X, features=features, method=method, kind=kind, grid_resolution=grid_resolution)
    pdp, axes = (result, result['grid_values'])
    expected_pdp_shape = (n_targets, *[grid_resolution for _ in range(len(features))])
    expected_ice_shape = (n_targets, n_instances, *[grid_resolution for _ in range(len(features))])
    if kind == 'average':
        assert pdp.average.shape == expected_pdp_shape
    elif kind == 'individual':
        assert pdp.individual.shape == expected_ice_shape
    else:
        assert pdp.average.shape == expected_pdp_shape
        assert pdp.individual.shape == expected_ice_shape
    expected_axes_shape = (len(features), grid_resolution)
    assert axes is not None
    assert np.asarray(axes).shape == expected_axes_shape