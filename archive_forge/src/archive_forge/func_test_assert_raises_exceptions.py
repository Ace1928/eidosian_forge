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
def test_assert_raises_exceptions():
    ir = IsotonicRegression()
    rng = np.random.RandomState(42)
    msg = 'Found input variables with inconsistent numbers of samples'
    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7, 3], [0.1, 0.6])
    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7])
    msg = 'X should be a 1d array'
    with pytest.raises(ValueError, match=msg):
        ir.fit(rng.randn(3, 10), [0, 1, 2])
    msg = 'Isotonic regression input X should be a 1d array'
    with pytest.raises(ValueError, match=msg):
        ir.transform(rng.randn(3, 10))