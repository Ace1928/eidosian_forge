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
@pytest.mark.parametrize('sample_weight', [False, True])
def test_enet_cv_grid_search(sample_weight):
    """Test that ElasticNetCV gives same result as GridSearchCV."""
    n_samples, n_features = (200, 10)
    cv = 5
    X, y = make_regression(n_samples=n_samples, n_features=n_features, effective_rank=10, n_informative=n_features - 4, noise=10, random_state=0)
    if sample_weight:
        sample_weight = np.linspace(1, 5, num=n_samples)
    else:
        sample_weight = None
    alphas = np.logspace(np.log10(1e-05), np.log10(1), num=10)
    l1_ratios = [0.1, 0.5, 0.9]
    reg = ElasticNetCV(cv=cv, alphas=alphas, l1_ratio=l1_ratios)
    reg.fit(X, y, sample_weight=sample_weight)
    param = {'alpha': alphas, 'l1_ratio': l1_ratios}
    gs = GridSearchCV(estimator=ElasticNet(), param_grid=param, cv=cv, scoring='neg_mean_squared_error').fit(X, y, sample_weight=sample_weight)
    assert reg.l1_ratio_ == pytest.approx(gs.best_params_['l1_ratio'])
    assert reg.alpha_ == pytest.approx(gs.best_params_['alpha'])