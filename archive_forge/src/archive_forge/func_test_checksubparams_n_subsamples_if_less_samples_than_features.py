import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import (
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
from sklearn.utils._testing import assert_almost_equal
def test_checksubparams_n_subsamples_if_less_samples_than_features():
    random_state = np.random.RandomState(0)
    n_samples, n_features = (10, 20)
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    theil_sen = TheilSenRegressor(n_subsamples=9, random_state=0)
    with pytest.raises(ValueError):
        theil_sen.fit(X, y)