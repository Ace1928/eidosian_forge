import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('EstimatorCV', [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV])
def test_cv_estimators_reject_params_with_no_routing_enabled(EstimatorCV):
    """Check that the models inheriting from class:`LinearModelCV` raise an
    error when any `params` are passed when routing is not enabled.
    """
    X, y = make_regression(random_state=42)
    groups = np.array([0, 1] * (len(y) // 2))
    estimator = EstimatorCV()
    msg = 'is only supported if enable_metadata_routing=True'
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, groups=groups)