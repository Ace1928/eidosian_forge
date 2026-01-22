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
def test_check_increasing_down():
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1.5, -2.77, -8.99, -8.99, -50]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert not is_increasing