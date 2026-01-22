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
@pytest.mark.parametrize('precompute, inner_precompute', [(True, True), ('auto', False), (False, False)])
def test_lassoCV_does_not_set_precompute(monkeypatch, precompute, inner_precompute):
    X, y, _, _ = build_dataset()
    calls = 0

    class LassoMock(Lasso):

        def fit(self, X, y):
            super().fit(X, y)
            nonlocal calls
            calls += 1
            assert self.precompute == inner_precompute
    monkeypatch.setattr('sklearn.linear_model._coordinate_descent.Lasso', LassoMock)
    clf = LassoCV(precompute=precompute)
    clf.fit(X, y)
    assert calls > 0