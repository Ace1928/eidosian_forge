import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_class_weights_cv():
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]
    reg = RidgeClassifierCV(class_weight=None, alphas=[0.01, 0.1, 1])
    reg.fit(X, y)
    reg = RidgeClassifierCV(class_weight={1: 0.001}, alphas=[0.01, 0.1, 1, 10])
    reg.fit(X, y)
    assert_array_equal(reg.predict([[-0.2, 2]]), np.array([-1]))