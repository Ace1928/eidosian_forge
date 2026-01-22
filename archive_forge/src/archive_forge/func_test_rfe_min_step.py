from operator import attrgetter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
def test_rfe_min_step(global_random_seed):
    n_features = 10
    X, y = make_friedman1(n_samples=50, n_features=n_features, random_state=global_random_seed)
    n_samples, n_features = X.shape
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, step=0.01)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2
    selector = RFE(estimator, step=0.2)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2
    selector = RFE(estimator, step=5)
    sel = selector.fit(X, y)
    assert sel.support_.sum() == n_features // 2