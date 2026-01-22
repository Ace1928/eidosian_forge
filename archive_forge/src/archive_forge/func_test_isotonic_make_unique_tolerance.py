import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_isotonic_make_unique_tolerance():
    X = np.array([0, 1, 1 + 1e-16, 2], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    ireg = IsotonicRegression().fit(X, y)
    y_pred = ireg.predict([0, 0.5, 1, 1.5, 2])
    assert_array_equal(y_pred, np.array([0, 0.75, 1.5, 2.25, 3]))
    assert_array_equal(ireg.X_thresholds_, np.array([0.0, 1.0, 2.0]))
    assert_array_equal(ireg.y_thresholds_, np.array([0.0, 1.5, 3.0]))