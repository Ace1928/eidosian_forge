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
@pytest.mark.parametrize('Estimator, data', [(LinearRegression, multioutput_regression_data), (LogisticRegression, binary_classification_data)])
def test_partial_dependence_equivalence_equal_sample_weight(Estimator, data):
    """Check that `sample_weight=None` is equivalent to having equal weights."""
    est = Estimator()
    (X, y), n_targets = data
    est.fit(X, y)
    sample_weight, params = (None, {'X': X, 'features': [1, 2], 'kind': 'average'})
    pdp_sw_none = partial_dependence(est, **params, sample_weight=sample_weight)
    sample_weight = np.ones(len(y))
    pdp_sw_unit = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none['average'], pdp_sw_unit['average'])
    sample_weight = 2 * np.ones(len(y))
    pdp_sw_doubling = partial_dependence(est, **params, sample_weight=sample_weight)
    assert_allclose(pdp_sw_none['average'], pdp_sw_doubling['average'])