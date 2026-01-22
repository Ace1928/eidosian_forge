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
def test_isotonic_regression_reversed():
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])
    y_ = IsotonicRegression(increasing=False).fit_transform(np.arange(len(y)), y)
    assert_array_equal(np.ones(y_[:-1].shape), y_[:-1] - y_[1:] >= 0)