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
@pytest.mark.parametrize('check_input', [True, False])
def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
    """Check that ElasticNet does not overwrite sample_weights."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    sample_weight_1_25 = 1.25 * np.ones_like(y)
    sample_weight = sample_weight_1_25.copy()
    reg = ElasticNet()
    reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)
    assert_array_equal(sample_weight, sample_weight_1_25)