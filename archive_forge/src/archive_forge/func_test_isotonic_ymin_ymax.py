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
def test_isotonic_ymin_ymax():
    x = np.array([1.263, 1.318, -0.572, 0.307, -0.707, -0.176, -1.599, 1.059, 1.396, 1.906, 0.21, 0.028, -0.081, 0.444, 0.018, -0.377, -0.896, -0.377, -1.327, 0.18])
    y = isotonic_regression(x, y_min=0.0, y_max=0.1)
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    y = isotonic_regression(x, y_min=0.0, y_max=0.1, increasing=False)
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    y = isotonic_regression(x, y_min=0.0, increasing=False)
    assert np.all(y >= 0)