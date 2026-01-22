import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_valid_n_bins():
    KBinsDiscretizer(n_bins=2).fit_transform(X)
    KBinsDiscretizer(n_bins=np.array([2])[0]).fit_transform(X)
    assert KBinsDiscretizer(n_bins=2).fit(X).n_bins_.dtype == np.dtype(int)