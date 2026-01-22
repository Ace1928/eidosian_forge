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
def test_isotonic_zero_weight_loop():
    rng = np.random.RandomState(42)
    regression = IsotonicRegression()
    n_samples = 50
    x = np.linspace(-3, 3, n_samples)
    y = x + rng.uniform(size=n_samples)
    w = rng.uniform(size=n_samples)
    w[5:8] = 0
    regression.fit(x, y, sample_weight=w)
    regression.fit(x, y, sample_weight=w)