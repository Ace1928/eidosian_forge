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
def test_check_increasing_small_number_of_samples():
    x = [0, 1, 2]
    y = [1, 1.1, 1.05]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert is_increasing