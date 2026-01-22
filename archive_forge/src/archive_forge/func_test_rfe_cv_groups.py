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
def test_rfe_cv_groups():
    generator = check_random_state(0)
    iris = load_iris()
    number_groups = 4
    groups = np.floor(np.linspace(0, number_groups, len(iris.target)))
    X = iris.data
    y = (iris.target > 0).astype(int)
    est_groups = RFECV(estimator=RandomForestClassifier(random_state=generator), step=1, scoring='accuracy', cv=GroupKFold(n_splits=2))
    est_groups.fit(X, y, groups=groups)
    assert est_groups.n_features_ > 0