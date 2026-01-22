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
def test_check_input_false():
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    X = check_array(X, order='F', dtype='float64')
    y = check_array(X, order='F', dtype='float64')
    clf = ElasticNet(selection='cyclic', tol=1e-08)
    clf.fit(X, y, check_input=False)
    X = check_array(X, order='F', dtype='float32')
    clf.fit(X, y, check_input=False)
    X = check_array(X, order='C', dtype='float64')
    with pytest.raises(ValueError):
        clf.fit(X, y, check_input=False)